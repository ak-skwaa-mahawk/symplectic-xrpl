use std::time::Instant;
use tokio::sync::mpsc;
use symplectic_test::coupled_lattice::LatticeEngine;
use symplectic_test::xrpl_feed::{start_xrpl_subscriber, XrplStreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Symplectic XRPL Headless Runner...");
    let (feed_tx, mut feed_rx) = mpsc::channel::<XrplStreamEvent>(2000);

    tokio::spawn(async move {
        start_xrpl_subscriber(feed_tx).await;
    });

    let mut lattice = LatticeEngine::new(8);
    let mut pending_txs: Vec<(u64, String)> = Vec::new();
    let mut last_ledger_instant = Instant::now();
    let mut total_admitted = 0u64;
    let mut total_rejected = 0u64;

    while let Some(ev) = feed_rx.recv().await {
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

                let batch_size = pending_txs.len();
                let mut admitted_in_batch = 0;
                let mut rejected_in_batch = 0;

                for (idx, (drops, _acc)) in pending_txs.drain(..).enumerate() {
                    match lattice.evaluate_admission(drops, idx) {
                        Ok(()) => {
                            total_admitted += 1;
                            admitted_in_batch += 1;
                        }
                        Err(_) => {
                            total_rejected += 1;
                            rejected_in_batch += 1;
                        }
                    }
                }

                lattice.step_symplectic(dt);
                lattice.update_phase_state();

                println!(
                    "Ledger #{:<9} | Network Txs: {:<4} | Buf: {:<4} | Adm: {:<4} ({}) | Rej: {:<4} ({}) | Energy: {:>8.2} / {:.0}",
                    ledger_index, tx_count, batch_size, admitted_in_batch, total_admitted, rejected_in_batch, total_rejected, lattice.total_energy(), lattice.e_max
                );
            }
        }
    }

    Ok(())
}
