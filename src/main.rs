use std::time::Instant;

const M: usize = 8;
const KAPPA: i64 = 1;
const E_MAX: i128 = 2_000_000_000;
const P_SITE_MAX: i64 = 30_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeState {
    pub q: [i64; M],
    pub p: [i64; M],
}

impl LatticeState {
    pub fn new() -> Self {
        Self { q: [0; M], p: [0; M] }
    }

    pub fn compute_energy(&self) -> i128 {
        let mut kinetic: i128 = 0;
        let mut potential: i128 = 0;
        for i in 0..M {
            kinetic += (self.p[i] as i128) * (self.p[i] as i128);
            potential += (self.q[i] as i128) * (self.q[i] as i128);
        }
        (kinetic + (KAPPA as i128) * potential) / 2
    }
}

pub fn forward_lifting_ntt(x: &mut [i64; M]) {
    let mut step = 1;
    while step < M {
        let jump = step * 2;
        for i in (0..M).step_by(jump) {
            for j in 0..step {
                let idx0 = i + j;
                let idx1 = i + j + step;
                let d = x[idx1].wrapping_sub(x[idx0]);
                let s = x[idx0].wrapping_add(d >> 1);
                x[idx0] = s;
                x[idx1] = d;
            }
        }
        step = jump;
    }
}

pub fn inverse_lifting_ntt(x: &mut [i64; M]) {
    let mut step = M / 2;
    while step >= 1 {
        let jump = step * 2;
        for i in (0..M).step_by(jump) {
            for j in 0..step {
                let idx0 = i + j;
                let idx1 = i + j + step;
                let s = x[idx0];
                let d = x[idx1];
                let x0 = s.wrapping_sub(d >> 1);
                let x1 = d.wrapping_add(x0);
                x[idx0] = x0;
                x[idx1] = x1;
            }
        }
        step /= 2;
    }
}

#[inline(always)]
pub fn symplectic_step(state: &mut LatticeState) {
    for i in 0..M {
        state.p[i] = state.p[i].wrapping_sub(KAPPA.wrapping_mul(state.q[i]));
        state.q[i] = state.q[i].wrapping_add(state.p[i]);
    }
}

#[inline(always)]
pub fn inverse_symplectic_step(state: &mut LatticeState) {
    for i in 0..M {
        state.q[i] = state.q[i].wrapping_sub(state.p[i]);
        state.p[i] = state.p[i].wrapping_add(KAPPA.wrapping_mul(state.q[i]));
    }
}

#[derive(Clone, Debug)]
pub struct Transaction {
    pub delta_p: [i64; M],
    pub utility: u64,
}

pub struct AdmissionResult {
    pub admitted_indices: Vec<usize>,
    pub p_rej: [i64; M],
}

fn compute_q_metric(p: &[i64; M], delta_p: &[i64; M]) -> i128 {
    let mut inner: i128 = 0;
    let mut norm_sq: i128 = 0;
    for i in 0..M {
        inner += (p[i] as i128) * (delta_p[i] as i128);
        norm_sq += (delta_p[i] as i128) * (delta_p[i] as i128);
    }
    2 * inner + norm_sq
}

pub fn process_batch(
    state: &mut LatticeState,
    batch: &[Transaction],
) -> AdmissionResult {
    let mut admitted = Vec::new();
    let mut p_rej = [0i64; M];

    let mut pass1 = Vec::new();
    let mut pass2 = Vec::new();

    for (idx, tx) in batch.iter().enumerate() {
        let q0 = compute_q_metric(&state.p, &tx.delta_p);
        if q0 <= 0 {
            pass1.push((idx, tx, q0));
        } else {
            pass2.push((idx, tx, q0));
        }
    }

    pass1.sort_by_key(|item| item.2);

    pass2.sort_by(|a, b| {
        let score_a = (a.1.utility as i128 * 1024) / (a.2.max(1));
        let score_b = (b.1.utility as i128 * 1024) / (b.2.max(1));
        score_b.cmp(&score_a)
    });

    for (orig_idx, tx, _) in pass1 {
        let q_k = compute_q_metric(&state.p, &tx.delta_p);
        if q_k <= 0 {
            for i in 0..M {
                state.p[i] = state.p[i].wrapping_add(tx.delta_p[i]);
            }
            admitted.push(orig_idx);
        } else {
            for i in 0..M {
                p_rej[i] = p_rej[i].wrapping_add(tx.delta_p[i]);
            }
        }
    }

    for (orig_idx, tx, _) in pass2 {
        let current_energy = state.compute_energy();
        let q_k = compute_q_metric(&state.p, &tx.delta_p);
        let predicted_energy = current_energy + (q_k / 2);

        let p_global = predicted_energy <= E_MAX;

        let mut p_local = true;
        for i in 0..M {
            let next_pi = state.p[i].wrapping_add(tx.delta_p[i]);
            if next_pi.abs() > P_SITE_MAX {
                p_local = false;
                break;
            }
        }

        if p_global && p_local {
            for i in 0..M {
                state.p[i] = state.p[i].wrapping_add(tx.delta_p[i]);
            }
            admitted.push(orig_idx);
        } else {
            for i in 0..M {
                p_rej[i] = p_rej[i].wrapping_add(tx.delta_p[i]);
            }
        }
    }

    AdmissionResult { admitted_indices: admitted, p_rej }
}

fn main() {
    println!("============================================================");
    println!("     INTEGER-N CANONICAL REFERENCE SIMULATOR (Z_2^64)       ");
    println!("============================================================");

    println!("\n[1] Verifying Reversible Lifting NTT Bijectivity...");
    let initial_signal: [i64; M] = [1054, -9823, 77123, -4, 553, -29841, 12, 908123];
    let mut signal = initial_signal;
    
    forward_lifting_ntt(&mut signal);
    inverse_lifting_ntt(&mut signal);
    
    assert_eq!(initial_signal, signal, "Lifting transform lost bits!");
    println!("    ✓ Bijectivity Confirmed: Forward -> Inverse returned identical bit pattern.");

    const EPOCH_STEPS: u64 = 10_000_000;
    println!("\n[2] Executing {} Symplectic Steps on Modular Ring...", EPOCH_STEPS);

    let mut state = LatticeState::new();
    state.q = [100, -50, 200, -150, 80, -30, 40, -10];
    state.p = [10, -20, 15, -5, 25, -15, 8, -12];

    let initial_state = state.clone();
    let initial_energy = state.compute_energy();

    let start = Instant::now();
    for _ in 0..EPOCH_STEPS {
        symplectic_step(&mut state);
    }
    let duration = start.elapsed();

    let end_energy = state.compute_energy();
    let drift = (end_energy - initial_energy).abs();

    println!("    Completed in: {:.2?}", duration);
    println!("    Throughput:   {:.2} Mstep/s", (EPOCH_STEPS as f64 / duration.as_secs_f64()) / 1e6);
    println!("    E(0):         {}", initial_energy);
    println!("    E(T):         {}", end_energy);
    println!("    Energy Drift: {} (Bounded within periodic orbit)", drift);

    print!("    Reversing {} steps... ", EPOCH_STEPS);
    for _ in 0..EPOCH_STEPS {
        inverse_symplectic_step(&mut state);
    }
    assert_eq!(state, initial_state, "Dynamical state did not return to origin!");
    println!("Done. Bit-exact return confirmed.");

    println!("\n[3] Testing Two-Pass Polar Admission & Bit-Exact Unwinding...");
    let transactions = vec![
        Transaction { delta_p: [-8, 15, -12, 4, -20, 10, -6, 10], utility: 50 },
        Transaction { delta_p: [-2, 5, -3, 1, -5, 5, -2, 2],      utility: 80 },
        Transaction { delta_p: [50, 50, 50, 50, 50, 50, 50, 50],   utility: 120 },
        Transaction { delta_p: [40_000, 0, 0, 0, 0, 0, 0, 0],      utility: 1000 },
        Transaction { delta_p: [10, -5, 12, -8, 6, -4, 2, -1],     utility: 300 },
    ];

    let pre_ingest_state = state.clone();
    let pre_energy = state.compute_energy();

    let result = process_batch(&mut state, &transactions);

    println!("    Transactions Admitted: {:?}", result.admitted_indices);
    println!("    Energy Pre-Ingest:     {}", pre_energy);
    println!("    Energy Post-Ingest:    {}", state.compute_energy());
    println!("    Rejected Momentum Buffer (p_rej): {:?}", result.p_rej);

    assert!(state.compute_energy() <= E_MAX, "Global headroom breached!");
    for p_i in state.p {
        assert!(p_i.abs() <= P_SITE_MAX, "Local site bound breached!");
    }
    println!("    ✓ Headroom and site bounds respected.");

    for &idx in result.admitted_indices.iter().rev() {
        let tx = &transactions[idx];
        for i in 0..M {
            state.p[i] = state.p[i].wrapping_sub(tx.delta_p[i]);
        }
    }
    assert_eq!(state, pre_ingest_state, "Unwind failed to match pre-ingest state!");
    println!("    ✓ Batch state unwound with zero bit loss (ΔS = 0).");
    println!("\nAll mathematical invariants validated under machine arithmetic.");
}
