use std::io;
use std::time::{Duration, Instant};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    widgets::{
        canvas::{Canvas, Points},
        BarChart, Block, Borders, List, ListItem, Sparkline, Paragraph,
    },
    Terminal,
};
use tokio::sync::mpsc;
use symplectic_test::coupled_lattice::LatticeEngine;
use symplectic_test::xrpl_feed::{start_xrpl_subscriber, XrplStreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (feed_tx, mut feed_rx) = mpsc::channel::<XrplStreamEvent>(2000);
    tokio::spawn(async move {
        start_xrpl_subscriber(feed_tx).await;
    });

    let mut lattice = LatticeEngine::new(8);
    let mut pending_txs: Vec<(u64, String)> = Vec::new();
    let mut event_logs: Vec<String> = Vec::new();
    let mut energy_history: Vec<u64> = vec![0; 40];
    let mut phase_trail: Vec<(f64, f64)> = Vec::new();

    let mut last_ledger_idx = 0u64;
    let mut last_ledger_instant = Instant::now();
    let mut current_tx_count = 0u32;
    let mut total_admitted = 0u64;
    let mut connected = false;

    'main_loop: loop {
        // Drain incoming messages
        while let Ok(ev) = feed_rx.try_recv() {
            connected = true;
            match ev {
                XrplStreamEvent::Tx { drops, account } => {
                    pending_txs.push((drops, account));
                }
                XrplStreamEvent::LedgerClosed { ledger_index, close_time_resolution, tx_count } => {
                    let now = Instant::now();
                    let elapsed = now.duration_since(last_ledger_instant).as_secs_f64();
                    last_ledger_instant = now;

                    let dt = if elapsed > 0.1 && elapsed < 10.0 {
                        0.05 * (elapsed / close_time_resolution as f64)
                    } else {
                        0.05
                    };

                    last_ledger_idx = ledger_index;
                    current_tx_count = tx_count;

                    for (idx, (drops, _acc)) in pending_txs.drain(..).enumerate() {
                        match lattice.evaluate_admission(drops, idx) {
                            Ok(()) => total_admitted += 1,
                            Err(e) => {
                                event_logs.push(format!("#{} {}", lattice.epoch_count, e));
                                if event_logs.len() > 10 {
                                    event_logs.remove(0);
                                }
                            }
                        }
                    }

                    lattice.step_symplectic(dt);
                    lattice.update_phase_state();

                    phase_trail.push((lattice.q[0], lattice.p[0]));
                    if phase_trail.len() > 80 {
                        phase_trail.remove(0);
                    }

                    energy_history.push((lattice.total_energy() / 10.0) as u64);
                    if energy_history.len() > 40 {
                        energy_history.remove(0);
                    }
                }
            }
        }

        let current_epoch = lattice.epoch_count;

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(10),
                    Constraint::Length(7),
                ])
                .split(f.area());

            let status_color = if connected { Color::Cyan } else { Color::Yellow };
            let top_text = if connected {
                format!(
                    " XRPL DYNAMIC PIPELINE | Ledger: #{} | Ledger Txs: {} | Epoch: #{} | Ingest Queue: {} | Total Adm: {} ",
                    last_ledger_idx, current_tx_count, current_epoch, pending_txs.len(), total_admitted
                )
            } else {
                " XRPL DYNAMIC PIPELINE | Connecting to wss://s1.ripple.com:51233... ".to_string()
            };

            let top_bar = Paragraph::new(top_text)
                .block(Block::default().borders(Borders::ALL))
                .style(Style::default().fg(status_color));
            f.render_widget(top_bar, chunks[0]);

            let mid_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[1]);

            let canvas = Canvas::default()
                .block(Block::default().borders(Borders::ALL).title("Phase Orbit (q0 vs p0)"))
                .x_bounds([-50.0, 50.0])
                .y_bounds([-1000.0, 1000.0])
                .paint(|ctx| {
                    ctx.draw(&Points {
                        coords: &phase_trail,
                        color: Color::Yellow,
                    });
                });
            f.render_widget(canvas, mid_chunks[0]);

            let bar_data: Vec<(&str, u64)> = lattice.p.iter().enumerate().map(|(i, &p)| {
                let label = match i {
                    0 => "S0", 1 => "S1", 2 => "S2", 3 => "S3",
                    4 => "S4", 5 => "S5", 6 => "S6", _ => "S7",
                };
                (label, p.abs() as u64)
            }).collect();

            let barchart = BarChart::default()
                .block(Block::default().borders(Borders::ALL).title("Site Momenta (p)"))
                .data(&bar_data)
                .bar_width(2)
                .bar_gap(1)
                .bar_style(Style::default().fg(Color::Green))
                .value_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));
            f.render_widget(barchart, mid_chunks[1]);

            let bot_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[2]);

            let sparkline = Sparkline::default()
                .block(Block::default().borders(Borders::ALL).title("Energy Trajectory (E/10)"))
                .style(Style::default().fg(Color::Magenta))
                .data(&energy_history)
                .bar_set(symbols::bar::NINE_LEVELS);
            f.render_widget(sparkline, bot_chunks[0]);

            let log_items: Vec<ListItem> = event_logs
                .iter()
                .map(|msg| ListItem::new(msg.as_str()).style(Style::default().fg(Color::Red)))
                .collect();
            let log_list = List::new(log_items)
                .block(Block::default().borders(Borders::ALL).title("Admission Gate Events"));
            f.render_widget(log_list, bot_chunks[1]);
        })?;

        // 30 FPS tick pacing and key event polling
        if event::poll(Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                        break 'main_loop;
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
