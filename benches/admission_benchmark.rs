use std::time::Instant;
use symplectic_test::coupled_lattice::LatticeEngine;

fn generate_synthetic_batch(size: usize, min_drops: u64, max_drops: u64) -> Vec<(u64, String)> {
    let mut batch = Vec::with_capacity(size);
    let mut rng_seed: u64 = 0x5EED_CAFE_DEAD_BEEF;

    for i in 0..size {
        // Fast Xorshift64* pseudo-random drop generator
        rng_seed ^= rng_seed >> 12;
        rng_seed ^= rng_seed << 25;
        rng_seed ^= rng_seed >> 27;
        let rand_val = rng_seed.wrapping_mul(0x2545F4914F6CDD1D);
        
        let drops = min_drops + (rand_val % (max_drops - min_drops + 1));
        let account = format!("rBenchAccount{:04}xxxx", i % 100);
        batch.push((drops, account));
    }
    batch
}

fn benchmark_batch(size: usize, batches_to_run: usize) {
    let mut lattice = LatticeEngine::new(8);
    let synthetic_txs = generate_synthetic_batch(size, 100, 10_000_000);

    let mut total_admitted = 0usize;
    let mut total_rejected = 0usize;

    println!("============================================================");
    println!(" Running Benchmark: Batch Size = {}, Iterations = {}", size, batches_to_run);
    println!("============================================================");

    let start_instant = Instant::now();

    for _ in 0..batches_to_run {
        for (idx, &(drops, _)) in synthetic_txs.iter().enumerate() {
            match lattice.evaluate_admission(drops, idx) {
                Ok(()) => total_admitted += 1,
                Err(_) => total_rejected += 1,
            }
        }

        // Advance the symplectic leapfrog integrator (dt = 0.05s) to allow dissipation
        lattice.step_symplectic(0.05);
        lattice.update_phase_state();
    }

    let elapsed = start_instant.elapsed();
    let total_txs = size * batches_to_run;
    let tps = total_txs as f64 / elapsed.as_secs_f64();
    let ns_per_tx = (elapsed.as_nanos() as f64) / (total_txs as f64);

    println!("Elapsed Time      : {:.3?}", elapsed);
    println!("Total Evaluations : {}", total_txs);
    println!("Admitted TXs      : {} ({:.2}%)", total_admitted, (total_admitted as f64 / total_txs as f64) * 100.0);
    println!("Rejected TXs      : {} ({:.2}%)", total_rejected, (total_rejected as f64 / total_txs as f64) * 100.0);
    println!("Throughput (TPS)  : {:>10.2} tx/s", tps);
    println!("Latency per Gate  : {:>10.2} ns/tx", ns_per_tx);
    println!("Final Energy      : {:.2} (E_max = {:.2})", lattice.total_energy(), lattice.e_max);
    println!();
}

fn main() {
    println!("\n=== SYMPLECTIC XRPL ADMISSION GATE BENCHMARK ===\n");
    
    // Warm-up run
    benchmark_batch(1_000, 10);

    // Standard high-load stress testing
    benchmark_batch(10_000, 20);
    benchmark_batch(50_000, 10);
    benchmark_batch(100_000, 5);
}
