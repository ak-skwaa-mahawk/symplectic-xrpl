use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use symplectic_test::{
    LatticeState, Transaction, RejectionCause, process_batch, symplectic_step,
    forward_lifting_ntt, inverse_lifting_ntt,
};

const M: usize = 8;
const P_MAX_KICK: i64 = 5_000;
const P_SITE_MAX: i64 = 30_000;
const V_SCALE: i64 = 1_000_000;
const WS_URL: &str = "wss://xrplcluster.com";
const EPOCH_WINDOW_MS: u64 = 500;

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
    let raw_mult = h1.finish() as usize;

    let mut h2 = DefaultHasher::new();
    epoch.hash(&mut h2);
    0xBF58476D1CE4E5B9u64.hash(&mut h2);
    let raw_trans = h2.finish() as usize;

    let a_e = (raw_mult << 1) | 1;
    let b_e = raw_trans % M;
    (a_e, b_e)
}

fn hash_to_site_salted(addr: &str, a_e: usize, b_e: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    addr.hash(&mut hasher);
    let base_hash = hasher.finish() as usize;
    (a_e.wrapping_mul(base_hash).wrapping_add(b_e)) % M
}

fn extract_drops(amount_val: &Option<Value>) -> u64 {
    match amount_val {
        Some(Value::String(drops_str)) => drops_str.parse::<u64>().unwrap_or(0),
        Some(Value::Object(map)) => {
            map.get("value")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .map(|f| (f * 1_000_000.0) as u64)
                .unwrap_or(0)
        }
        _ => 0,
    }
}

/// Evaluates dynamic dyadic shift ceil(log2((B * P_MAX_KICK) / P_SITE_MAX))
#[inline(always)]
pub fn compute_dynamic_shift(batch_size: usize) -> usize {
    const BASE_SHIFT: usize = 2; // Default 75% attenuation (shift by 2)
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

    // Apply modal filtering with the batch-scaled shift
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Connecting to live XRPL node: {}", WS_URL);
    let (ws_stream, _) = connect_async(WS_URL).await?;
    println!("✓ WebSocket Handshake successful.");

    let (mut write, mut read) = ws_stream.split();

    let subscribe_cmd = json!({
        "command": "subscribe",
        "streams": ["transactions"]
    });
    write.send(Message::Text(subscribe_cmd.to_string().into())).await?;
    println!("✓ Subscribed to proposed transaction stream.");

    let (tx_chan, mut rx_chan) = mpsc::channel::<XrplTxEvent>(4096);

    tokio::spawn(async move {
        while let Some(msg_res) = read.next().await {
            if let Ok(Message::Text(raw_utf8)) = msg_res {
                if let Ok(envelope) = serde_json::from_str::<XrplTxEnvelope>(raw_utf8.as_str()) {
                    if let Some(tx) = envelope.transaction {
                        if tx.tx_type.as_deref() == Some("Payment") {
                            let _ = tx_chan.send(tx).await;
                        }
                    }
                }
            }
        }
    });

    println!("\nIngesting epochs with Dynamic Modal Attenuation ({} ms window)...\n", EPOCH_WINDOW_MS);
    let mut lattice = LatticeState::new();
    let mut epoch_ticker = interval(Duration::from_millis(EPOCH_WINDOW_MS));
    let mut epoch_counter: u64 = 0;
    let mut raw_event_buffer = Vec::new();

    loop {
        epoch_ticker.tick().await;

        while let Ok(event) = rx_chan.try_recv() {
            raw_event_buffer.push(event);
        }

        epoch_counter += 1;
        let e_pre = lattice.compute_energy();

        if !raw_event_buffer.is_empty() {
            let batch_size = raw_event_buffer.len();
            let shift = compute_dynamic_shift(batch_size);

            let batch: Vec<Transaction> = raw_event_buffer
                .iter()
                .filter_map(|ev| xrpl_to_lattice_impulse_dynamic(ev, epoch_counter, shift))
                .collect();

            let res = process_batch(&mut lattice, &batch);
            let admitted_cnt = res.admitted_indices.len();
            let rejected_cnt = res.rejections.len();

            symplectic_step(&mut lattice);
            let e_post = lattice.compute_energy();

            println!(
                "Epoch #{:05} | B: {:3} | Shift: >>{} | Admitted: {:3} | Rejected: {:3} | E_pre: {:10} | E_post: {:10}",
                epoch_counter, batch_size, shift, admitted_cnt, rejected_cnt, e_pre, e_post
            );

            if !res.rejections.is_empty() {
                let mut p1_flips = 0;
                let mut global_overflows = 0;
                let mut local_spikes = 0;
                let mut dual_fails = 0;

                for rej in &res.rejections {
                    match &rej.cause {
                        RejectionCause::Pass1MonotonicityFlip { .. } => p1_flips += 1,
                        RejectionCause::GlobalHeadroomExceeded { .. } => global_overflows += 1,
                        RejectionCause::LocalSiteBoundExceeded { .. } => local_spikes += 1,
                        RejectionCause::DualGlobalAndLocal { .. } => dual_fails += 1,
                    }
                }

                println!(
                    "  └─ Gate: [P1 Flips: {} | Headroom: {} | L_inf: {} | Dual: {}]",
                    p1_flips, global_overflows, local_spikes, dual_fails
                );
            }

            print!("  └─ Sites |p_i|: [");
            for (idx, p_i) in lattice.p.iter().enumerate() {
                print!("S{}: {:5}{}", idx, p_i.abs(), if idx + 1 < M { ", " } else { "" });
            }
            println!("]");

            raw_event_buffer.clear();
        } else {
            symplectic_step(&mut lattice);
            let e_post = lattice.compute_energy();
            println!(
                "Epoch #{:05} | Idle   | Shift: --  | Admitted:   0 | Rejected:   0 | E_pre: {:10} | E_post: {:10}",
                epoch_counter, e_pre, e_post
            );
        }
    }
}
