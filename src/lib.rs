pub const M: usize = 8;
pub const KAPPA: i64 = 1;
pub const E_MAX: i128 = 2_000_000_000;
pub const P_SITE_MAX: i64 = 30_000;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RejectionCause {
    Pass1MonotonicityFlip { q_k: i128 },
    GlobalHeadroomExceeded { projected_energy: i128, e_max: i128 },
    LocalSiteBoundExceeded { site: usize, projected_pi: i64, limit: i64 },
    DualGlobalAndLocal { projected_energy: i128, site: usize, projected_pi: i64 },
}

#[derive(Clone, Debug)]
pub struct RejectedRecord {
    pub original_idx: usize,
    pub cause: RejectionCause,
    pub delta_p: [i64; M],
    pub utility: u64,
}

pub struct AdmissionResult {
    pub admitted_indices: Vec<usize>,
    pub p_rej: [i64; M],
    pub rejections: Vec<RejectedRecord>,
}

pub fn compute_q_metric(p: &[i64; M], delta_p: &[i64; M]) -> i128 {
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
    let mut rejections = Vec::new();
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

    // Pass 1: Strict Contraction Check
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
            rejections.push(RejectedRecord {
                original_idx: orig_idx,
                cause: RejectionCause::Pass1MonotonicityFlip { q_k },
                delta_p: tx.delta_p,
                utility: tx.utility,
            });
        }
    }

    // Pass 2: Dual-Invariant Gate Check
    for (orig_idx, tx, _) in pass2 {
        let current_energy = state.compute_energy();
        let q_k = compute_q_metric(&state.p, &tx.delta_p);
        let predicted_energy = current_energy + (q_k / 2);

        let p_global = predicted_energy <= E_MAX;

        let mut site_failure: Option<(usize, i64)> = None;
        for i in 0..M {
            let next_pi = state.p[i].wrapping_add(tx.delta_p[i]);
            if next_pi.abs() > P_SITE_MAX {
                site_failure = Some((i, next_pi));
                break;
            }
        }

        match (p_global, site_failure) {
            (true, None) => {
                for i in 0..M {
                    state.p[i] = state.p[i].wrapping_add(tx.delta_p[i]);
                }
                admitted.push(orig_idx);
            }
            (false, None) => {
                for i in 0..M {
                    p_rej[i] = p_rej[i].wrapping_add(tx.delta_p[i]);
                }
                rejections.push(RejectedRecord {
                    original_idx: orig_idx,
                    cause: RejectionCause::GlobalHeadroomExceeded {
                        projected_energy: predicted_energy,
                        e_max: E_MAX,
                    },
                    delta_p: tx.delta_p,
                    utility: tx.utility,
                });
            }
            (true, Some((site, pi))) => {
                for i in 0..M {
                    p_rej[i] = p_rej[i].wrapping_add(tx.delta_p[i]);
                }
                rejections.push(RejectedRecord {
                    original_idx: orig_idx,
                    cause: RejectionCause::LocalSiteBoundExceeded {
                        site,
                        projected_pi: pi,
                        limit: P_SITE_MAX,
                    },
                    delta_p: tx.delta_p,
                    utility: tx.utility,
                });
            }
            (false, Some((site, pi))) => {
                for i in 0..M {
                    p_rej[i] = p_rej[i].wrapping_add(tx.delta_p[i]);
                }
                rejections.push(RejectedRecord {
                    original_idx: orig_idx,
                    cause: RejectionCause::DualGlobalAndLocal {
                        projected_energy: predicted_energy,
                        site,
                        projected_pi: pi,
                    },
                    delta_p: tx.delta_p,
                    utility: tx.utility,
                });
            }
        }
    }

    AdmissionResult { admitted_indices: admitted, p_rej, rejections }
}
