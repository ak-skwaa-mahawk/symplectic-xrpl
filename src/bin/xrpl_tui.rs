use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::io::stdout;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::{SinkExt, StreamExt};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{
        canvas::{Canvas, Line},
        Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline,
    },
    Terminal,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use symplectic_test::{
    forward_lifting_ntt, inverse_lifting_ntt, process_batch, symplectic_step,
    LatticeState, RejectionCause, Transaction, M, P_SITE_MAX,
};

const P_MAX_KICK: i64 = 5_000;
const V_SCALE: i64 = 1_000_000;
const WS_URL: &str = "wss://xrplcluster.com";
const EPOCH_WINDOW_MS: u64 = 250;

#[derive(Clone, Debug, Deserialize)]
pub struct XrplTxEnvelope {
    pub transaction: Option<XrplTxEvent>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct XrplTxEvent {
    #[serde(rename = "Account")]
    pub account: Option<String>,
    #[serde(rename = "Destination")]
    pub destination: Option<String>,
    #[serde(rename = "Fee")]
    pub fee: Option<String>,
    #[serde(rename = "Amount")]
    pub amount: Option<Value>,
    #[serde(rename = "TransactionType")]
    pub tx_type: Option<String>,
}

fn derive_epoch_permutation_params(epoch: u64) -> (usize, usize) {
    let mut h1 = DefaultHasher::new();
    epoch.hash(&mut h1);
    0x9E3779B97F4A7C15u64.hash(&mut h1);
    let a_e = ((h1.finish() as usize) << 1) | 1;

    let mut h2 = DefaultHasher::new();
    epoch.hash(&mut h2);
    0xBF58476D1CE4E5B9u64.hash(&mut h2);
    let b_e = (h2.finish() as usize) % M;

    (a_e, b_e)
}

fn hash_to_site_salted(addr: &str, a_e: usize, b_e: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    addr.hash(&mut hasher);
    (a_e.wrapping_mul(hasher.finish() as usize).wrapping_add(b_e)) % M
}

fn extract_drops(amount_val: &Option<Value>) -> u64 {
    match amount_val {
        Some(Value::String(drops_str)) => drops_str.parse::<u64>().unwrap_or(0),
        Some(Value::Object(map)) => map
            .get("value")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .map(|f| (f * 1_000_000.0) as u64)
            .unwrap_or(0),
        _ => 0,
    }
}

pub fn compute_dynamic_shift(batch_size: usize) -> usize {
    const BASE_SHIFT: usize = 2;
    if batch_size <= 6 {
        return BASE_SHIFT;
    }
    let numerator = (batch_size as u64) * (P_MAX_KICK as u64);
    let ratio = (numerator + (P_SITE_MAX as u64 - 1)) / (P_SITE_MAX as u64);
    let dynamic_shift = if ratio <= 1 {
        0
    } else {
        (64 - (ratio - 1).leading_zeros()) as usize
    };
    dynamic_shift.max(BASE_SHIFT)
}

pub fn xrpl_to_lattice_impulse_dynamic(
    tx: &XrplTxEvent,
    epoch: u64,
    shift: usize,
) -> Option<Transaction> {
    let src = tx.account.as_deref()?;
    let dst = tx.destination.as_deref()?;
    let raw_amount = extract_drops(&tx.amount);
    let fee_drops = tx.fee.as_deref().and_then(|f| f.parse::<u64>().ok()).unwrap_or(12);

    if raw_amount == 0 {
        return None;
    }

    let (a_e, b_e) = derive_epoch_permutation_params(epoch);
    let src_node = hash_to_site_salted(src, a_e, b_e);
    let mut dst_node = hash_to_site_salted(dst, a_e, b_e);

    if src_node == dst_node {
        dst_node = (src_node + a_e) % M;
    }

    let u_abs = raw_amount.min(i64::MAX as u64) as i64;
    let sat_impulse = (u_abs * P_MAX_KICK) / (u_abs + V_SCALE);

    let mut dipole = [0i64; M];
    dipole[src_node] = sat_impulse;
    dipole[dst_node] = -sat_impulse;

    let mut modal = dipole;
    forward_lifting_ntt(&mut modal);

    for k in (M / 2)..M {
        modal[k] >>= shift;
    }

    inverse_lifting_ntt(&mut modal);

    Some(Transaction {
        delta_p: modal,
        utility: fee_drops,
    })
}

#[derive(Clone, Debug)]
pub struct EpochTelemetry {
    pub epoch: u64,
    pub batch_size: usize,
    pub admitted: usize,
    pub rejected: usize,
    pub energy: i128,
    pub q0: i64,
    pub p0: i64,
    pub site_momenta: [i64; M],
    pub log_entry: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (ui_tx, mut ui_rx) = mpsc::channel::<EpochTelemetry>(128);

    tokio::spawn(async move {
        let ws_stream = match connect_async(WS_URL).await {
            Ok((s, _)) => s,
            Err(e) => {
                eprintln!("WS Err: {e}");
                return;
            }
        };
        let (mut write, mut read) = ws_stream.split();

        let subscribe_cmd = json!({
            "command": "subscribe",
            "streams": ["transactions"]
        });
        if write.send(Message::Text(subscribe_cmd.to_string().into())).await.is_err() {
            return;
        }

        let (tx_chan, mut rx_chan) = mpsc::channel::<XrplTxEvent>(4096);

        tokio::spawn(async move {
            while let Some(Ok(Message::Text(raw_utf8))) = read.next().await {
                if let Ok(env) = serde_json::from_str::<XrplTxEnvelope>(raw_utf8.as_str()) {
                    if let Some(tx) = env.transaction {
                        if tx.tx_type.as_deref() == Some("Payment") {
                            let _ = tx_chan.send(tx).await;
                        }
                    }
                }
            }
        });

        let mut lattice = LatticeState::new();
        let mut epoch_ticker = interval(Duration::from_millis(EPOCH_WINDOW_MS));
        let mut epoch_counter: u64 = 0;
        let mut raw_buffer = Vec::new();

        loop {
            epoch_ticker.tick().await;

            while let Ok(event) = rx_chan.try_recv() {
                raw_buffer.push(event);
            }

            epoch_counter += 1;
            let batch_size = raw_buffer.len();
            let shift = compute_dynamic_shift(batch_size);

            let (admitted, rejected, log_entry) = if !raw_buffer.is_empty() {
                let batch: Vec<Transaction> = raw_buffer
                    .iter()
                    .filter_map(|ev| xrpl_to_lattice_impulse_dynamic(ev, epoch_counter, shift))
                    .collect();

                let res = process_batch(&mut lattice, &batch);
                let adm = res.admitted_indices.len();
                let rej = res.rejections.len();

                let log = if let Some(first) = res.rejections.first() {
                    match &first.cause {
                        RejectionCause::Pass1MonotonicityFlip { q_k } => {
                            Some(format!("P1 Flip: Q={}", q_k))
                        }
                        RejectionCause::LocalSiteBoundExceeded { site, projected_pi, .. } => {
                            Some(format!("L_inf: S[{}]={}", site, projected_pi.abs()))
                        }
                        RejectionCause::GlobalHeadroomExceeded { projected_energy, .. } => {
                            Some(format!("E_max: {}", projected_energy / 1_000_000))
                        }
                        RejectionCause::DualGlobalAndLocal { site, .. } => {
                            Some(format!("Dual Breach: S[{}]", site))
                        }
                    }
                } else {
                    None
                };

                raw_buffer.clear();
                (adm, rej, log)
            } else {
                (0, 0, None)
            };

            symplectic_step(&mut lattice);

            let mut site_momenta = [0i64; M];
            for i in 0..M {
                site_momenta[i] = lattice.p[i].abs();
            }

            let telem = EpochTelemetry {
                epoch: epoch_counter,
                batch_size,
                admitted,
                rejected,
                energy: lattice.compute_energy(),
                q0: lattice.q[0],
                p0: lattice.p[0],
                site_momenta,
                log_entry,
            };

            if ui_tx.send(telem).await.is_err() {
                break;
            }
        }
    });

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut orbit_trail: VecDeque<(f64, f64)> = VecDeque::with_capacity(256);
    let mut energy_history: Vec<u64> = vec![0; 60];
    let mut event_logs: VecDeque<String> = VecDeque::with_capacity(30);
    let mut latest_telem: Option<EpochTelemetry> = None;

    let run_res = loop {
        while let Ok(telem) = ui_rx.try_recv() {
            orbit_trail.push_back((telem.q0 as f64, telem.p0 as f64));
            if orbit_trail.len() > 200 {
                orbit_trail.pop_front();
            }

            let scaled_e = (telem.energy / 10_000_000).max(0) as u64;
            energy_history.remove(0);
            energy_history.push(scaled_e);

            if let Some(msg) = telem.log_entry.clone() {
                event_logs.push_back(format!("#{}: {}", telem.epoch, msg));
                if event_logs.len() > 15 {
                    event_logs.pop_front();
                }
            }

            latest_telem = Some(telem);
        }

        // Catch drawing errors gracefully rather than immediately returning
        let draw_status = terminal.draw(|f| {
            let area = f.area();
            if area.width < 20 || area.height < 10 {
                return;
            }

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(8),
                    Constraint::Length(6),
                ])
                .split(area);

            let header_text = if let Some(ref t) = latest_telem {
                format!(
                    " XRPL INTEGER-N PIPELINE | #{} | Ingest: {} | Adm: {} | Rej: {} | E: {} ",
                    t.epoch, t.batch_size, t.admitted, t.rejected, t.energy
                )
            } else {
                " Connecting to XRPL Mainnet (wss://xrplcluster.com)...".to_string()
            };
            let header = Paragraph::new(header_text)
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .block(Block::default().borders(Borders::ALL).title("Consensus Status"));
            f.render_widget(header, chunks[0]);

            let middle_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[1]);

            let canvas = Canvas::default()
                .block(Block::default().borders(Borders::ALL).title("Phase Torus (q0 vs p0)"))
                .x_bounds([-40_000.0, 40_000.0])
                .y_bounds([-40_000.0, 40_000.0])
                .paint(|ctx| {
                    if orbit_trail.len() > 1 {
                        for i in 0..(orbit_trail.len() - 1) {
                            let (x1, y1) = orbit_trail[i];
                            let (x2, y2) = orbit_trail[i + 1];
                            ctx.draw(&Line {
                                x1,
                                y1,
                                x2,
                                y2,
                                color: Color::Yellow,
                            });
                        }
                    }
                });
            f.render_widget(canvas, middle_chunks[0]);

            let gauge_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Ratio(1, 8),
                    Constraint::Ratio(1, 8),
                    Constraint::Ratio(1, 8),
                    Constraint::Ratio(1, 8),
                    Constraint::Ratio(1, 8),
                    Constraint::Ratio(1, 8),
                    Constraint::Ratio(1, 8),
                    Constraint::Ratio(1, 8),
                ])
                .split(middle_chunks[1]);

            for i in 0..M {
                let val = latest_telem
                    .as_ref()
                    .map(|t| t.site_momenta[i])
                    .unwrap_or(0);
                let pct = ((val as f64 / P_SITE_MAX as f64) * 100.0).clamp(0.0, 100.0) as u16;
                let color = if pct > 85 {
                    Color::Red
                } else if pct > 50 {
                    Color::Yellow
                } else {
                    Color::Green
                };
                let gauge = Gauge::default()
                    .block(Block::default().borders(Borders::NONE))
                    .gauge_style(Style::default().fg(color))
                    .percent(pct)
                    .label(format!("S{}: {} / {}", i, val, P_SITE_MAX));
                f.render_widget(gauge, gauge_chunks[i]);
            }

            let bottom_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
                .split(chunks[2]);

            let sparkline = Sparkline::default()
                .block(Block::default().borders(Borders::ALL).title("Energy (E / 10M)"))
                .data(&energy_history)
                .style(Style::default().fg(Color::Magenta));
            f.render_widget(sparkline, bottom_chunks[0]);

            let log_items: Vec<ListItem> = event_logs
                .iter()
                .map(|msg| ListItem::new(msg.as_str()).style(Style::default().fg(Color::Red)))
                .collect();
            let log_list = List::new(log_items)
                .block(Block::default().borders(Borders::ALL).title("Gate Events"));
            f.render_widget(log_list, bottom_chunks[1]);
        });

        if let Err(e) = draw_status {
            eprintln!("Draw warning: {e}");
        }

        // Only break on valid interactive key press
        if let Ok(true) = event::poll(Duration::from_millis(30)) {
            if let Ok(Event::Key(key)) = event::read() {
                if key.kind == KeyEventKind::Press {
                    if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                        break Ok(());
                    }
                }
            }
        }
    };

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    run_res
}
