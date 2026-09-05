use std::f64::consts::PI;
use symplectic_test::coupled_lattice::LatticeEngine;

/// Pillar 1 & 4: Symplectic Jacobian determinant det(J) == 1 in the non-dissipative limit (D = 0)
#[test]
fn test_symplectic_volume_preservation() {
    let mut lattice = LatticeEngine::new(4);
    lattice.d_damping = 0.0; // Conserved Hamiltonian regime
    lattice.q = vec![0.5, -0.2, 0.1, -0.4];
    lattice.p = vec![1.2, 0.8, -0.5, 0.3];

    let dt = 0.01;
    let eps = 1e-6;
    let dim = 8; // 4 positions + 4 momenta

    // Construct the 8x8 numerical Jacobian matrix J_ij = d(x_next_i) / d(x_prev_j)
    let mut j_matrix = vec![vec![0.0; dim]; dim];

    for j in 0..dim {
        let mut forward_state = lattice.clone_state();
        let mut backward_state = lattice.clone_state();

        if j < 4 {
            forward_state.0[j] += eps;
            backward_state.0[j] -= eps;
        } else {
            forward_state.1[j - 4] += eps;
            backward_state.1[j - 4] -= eps;
        }

        let mut lat_f = LatticeEngine::new(4);
        lat_f.d_damping = 0.0;
        lat_f.q = forward_state.0;
        lat_f.p = forward_state.1;
        lat_f.step_symplectic(dt);

        let mut lat_b = LatticeEngine::new(4);
        lat_b.d_damping = 0.0;
        lat_b.q = backward_state.0;
        lat_b.p = backward_state.1;
        lat_b.step_symplectic(dt);

        for i in 0..4 {
            j_matrix[i][j] = (lat_f.q[i] - lat_b.q[i]) / (2.0 * eps);
            j_matrix[i + 4][j] = (lat_f.p[i] - lat_b.p[i]) / (2.0 * eps);
        }
    }

    let det = determinant_8x8(&j_matrix);
    assert!(
        (det - 1.0).abs() < 1e-5,
        "Symplectic volume failure: det(J) = {:.8} != 1.0",
        det
    );
}

/// Pillar 4: Lyapunov strict dissipation under zero input (dH/dt <= 0)
#[test]
fn test_lyapunov_dissipative_monotonicity() {
    let mut lattice = LatticeEngine::new(8);
    lattice.d_damping = 0.1; // Active Rayleigh dissipation
    lattice.q = vec![1.0, -0.5, 0.3, 0.8, -0.2, 0.4, -0.7, 0.1];
    lattice.p = vec![5.0, -3.0, 2.0, 4.0, -1.0, 3.0, -2.0, 1.5];

    let mut prev_energy = lattice.total_energy();

    for _ in 0..100 {
        lattice.step_symplectic(0.02);
        let curr_energy = lattice.total_energy();

        assert!(
            curr_energy <= prev_energy + 1e-9,
            "Lyapunov violation: Energy increased from {:.6} to {:.6}",
            prev_energy,
            curr_energy
        );
        prev_energy = curr_energy;
    }
}

/// Pillar 3: Dynamic phase recurrence divergence & non-convergence to static float 3.1730059
#[test]
fn test_phase_recurrence_non_convergence() {
    let mut b_n = 0.0f64;
    let h = 1.5f64;
    let target_float = 3.1730059f64;
    let mut unmodulated_sum = 0.0f64;

    for n in 2..=10_000 {
        let step = h * ((n as f64).ln() / (n as f64));
        unmodulated_sum += step;
        b_n = (b_n + step) % (2.0 * PI);

        // Step size must monotonically shrink toward zero
        let next_step = h * (((n + 1) as f64).ln() / ((n + 1) as f64));
        if n > 3 {
            assert!(next_step < step, "Step size is not monotonically decreasing at n={}", n);
        }
    }

    // Unmodulated length must diverge
    assert!(unmodulated_sum > 10.0, "Sum of (ln n)/n should diverge");

    // The sequence is dynamical and does not settle on the float literal
    assert!(
        (b_n - target_float).abs() > 1e-4,
        "Recurrence state unlawfully settled on literal {}: b_n = {}",
        target_float,
        b_n
    );
}

/// Pillar 2: Curvature scale collapse test (ratio depends on r/R)
#[test]
fn test_hyperbolic_scale_dependence() {
    let hyperbolic_ratio = |x: f64| -> f64 { PI * x.sinh() / x };

    let x0 = 0.24458;
    let val_at_x0 = hyperbolic_ratio(x0);
    assert!((val_at_x0 - 3.17300858).abs() < 1e-4);

    // Scale dilation r -> 2r
    let x_dilated = 2.0 * x0;
    let val_dilated = hyperbolic_ratio(x_dilated);

    assert!(
        (val_dilated - val_at_x0).abs() > 0.01,
        "Hyperbolic ratio must depend on scale: f(x0) = {}, f(2x0) = {}",
        val_at_x0,
        val_dilated
    );
}

// Helper trait to allow state cloning for perturbation tests
trait StateClone {
    fn clone_state(&self) -> (Vec<f64>, Vec<f64>);
}

impl StateClone for LatticeEngine {
    fn clone_state(&self) -> (Vec<f64>, Vec<f64>) {
        (self.q.clone(), self.p.clone())
    }
}

// Numerical 8x8 determinant via Gaussian elimination
fn determinant_8x8(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut det = 1.0;

    for i in 0..n {
        let mut pivot = i;
        for j in (i + 1)..n {
            if a[j][i].abs() > a[pivot][i].abs() {
                pivot = j;
            }
        }

        if a[pivot][i].abs() < 1e-12 {
            return 0.0;
        }

        if i != pivot {
            a.swap(i, pivot);
            det = -det;
        }

        det *= a[i][i];

        for j in (i + 1)..n {
            let factor = a[j][i] / a[i][i];
            for k in (i + 1)..n {
                a[j][k] -= factor * a[i][k];
            }
        }
    }

    det
}
