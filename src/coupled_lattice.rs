use crate::SymplecticDDS;

pub struct CoupledLattice {
    pub node_a: SymplecticDDS,
    pub node_b: SymplecticDDS,
    pub coupling_strength: i64, // Scaling factor for mutual spring
}

impl CoupledLattice {
    pub fn new(tuning_a: u32, tuning_b: u32, coupling_strength: i64) -> Self {
        Self {
            node_a: SymplecticDDS::new(tuning_a),
            node_b: SymplecticDDS::new(tuning_b),
            coupling_strength,
        }
    }

    #[inline(always)]
    fn coupling_force(&self, delta_theta: u32) -> i64 {
        let signed_delta = (delta_theta as i32) as i64;
        let base_force = if signed_delta.abs() < (1 << 16) {
            (3i64 * signed_delta) / 65536
        } else {
            let idx = (delta_theta >> 16) as usize;
            self.node_a.lut[idx] as i64
        };
        (base_force * self.coupling_strength) / 100
    }

    pub fn step_forward(&mut self, u_a: i64, u_b: i64) {
        // 1. Half-kick from self-restoring force and external driving
        let f_self_a1 = u_a - self.node_a.restoring_force(self.node_a.theta);
        let f_self_b1 = u_b - self.node_b.restoring_force(self.node_b.theta);
        let mut p_a_half = self.node_a.p + (f_self_a1 >> 1);
        let mut p_b_half = self.node_b.p + (f_self_b1 >> 1);

        // 2. Mutual coupling half-kick
        let d_theta_1 = self.node_b.theta.wrapping_sub(self.node_a.theta);
        let f_c1 = self.coupling_force(d_theta_1);
        p_a_half += f_c1 >> 1;
        p_b_half -= f_c1 >> 1;

        // 3. Phase drift step
        let delta_a = self.node_a.tuning_word.wrapping_add((p_a_half >> SymplecticDDS::SHIFT) as u32);
        let delta_b = self.node_b.tuning_word.wrapping_add((p_b_half >> SymplecticDDS::SHIFT) as u32);
        self.node_a.theta = self.node_a.theta.wrapping_add(delta_a);
        self.node_b.theta = self.node_b.theta.wrapping_add(delta_b);

        // 4. Mutual coupling second half-kick
        let d_theta_2 = self.node_b.theta.wrapping_sub(self.node_a.theta);
        let f_c2 = self.coupling_force(d_theta_2);
        let p_a_mid = p_a_half + (f_c2 >> 1);
        let p_b_mid = p_b_half - (f_c2 >> 1);

        // 5. Self-restoring second half-kick
        let f_self_a2 = u_a - self.node_a.restoring_force(self.node_a.theta);
        let f_self_b2 = u_b - self.node_b.restoring_force(self.node_b.theta);
        self.node_a.p = p_a_mid + (f_self_a2 >> 1);
        self.node_b.p = p_b_mid + (f_self_b2 >> 1);
    }

    pub fn step_backward(&mut self, u_a: i64, u_b: i64) {
        // Reverse Step 5: Undo self-restoring second half-kick
        let f_self_a2 = u_a - self.node_a.restoring_force(self.node_a.theta);
        let f_self_b2 = u_b - self.node_b.restoring_force(self.node_b.theta);
        let p_a_mid = self.node_a.p - (f_self_a2 >> 1);
        let p_b_mid = self.node_b.p - (f_self_b2 >> 1);

        // Reverse Step 4: Undo mutual coupling second half-kick
        let d_theta_2 = self.node_b.theta.wrapping_sub(self.node_a.theta);
        let f_c2 = self.coupling_force(d_theta_2);
        let p_a_half = p_a_mid - (f_c2 >> 1);
        let p_b_half = p_b_mid + (f_c2 >> 1);

        // Reverse Step 3: Undo phase drift
        let delta_a = self.node_a.tuning_word.wrapping_add((p_a_half >> SymplecticDDS::SHIFT) as u32);
        let delta_b = self.node_b.tuning_word.wrapping_add((p_b_half >> SymplecticDDS::SHIFT) as u32);
        self.node_a.theta = self.node_a.theta.wrapping_sub(delta_a);
        self.node_b.theta = self.node_b.theta.wrapping_sub(delta_b);

        // Reverse Step 2: Undo mutual coupling first half-kick
        let d_theta_1 = self.node_b.theta.wrapping_sub(self.node_a.theta);
        let f_c1 = self.coupling_force(d_theta_1);
        let p_a_start = p_a_half - (f_c1 >> 1);
        let p_b_start = p_b_half + (f_c1 >> 1);

        // Reverse Step 1: Undo self-restoring first half-kick
        let f_self_a1 = u_a - self.node_a.restoring_force(self.node_a.theta);
        let f_self_b1 = u_b - self.node_b.restoring_force(self.node_b.theta);
        self.node_a.p = p_a_start - (f_self_a1 >> 1);
        self.node_b.p = p_b_start - (f_self_b1 >> 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupled_energy_beat_and_exact_reversal() {
        // Initialize Node A excited, Node B at rest
        let mut lattice = CoupledLattice::new(0, 0, 50);
        lattice.node_a.p = 500_000;
        lattice.node_b.p = 0;

        let initial_state = (
            lattice.node_a.theta,
            lattice.node_a.p,
            lattice.node_b.theta,
            lattice.node_b.p,
        );

        let steps = 150_000;
        let mut inputs = Vec::with_capacity(steps);

        let mut node_b_absorbed_energy = false;

        for step in 0..steps {
            let u_a = (((step % 89) as i64) - 44) * 2;
            let u_b = (((step % 97) as i64) - 48) * 2;
            inputs.push((u_a, u_b));

            lattice.step_forward(u_a, u_b);

            // Verify that energy transfers across the coupling to Node B
            if lattice.node_b.p.abs() > 100_000 {
                node_b_absorbed_energy = true;
            }
        }

        assert!(node_b_absorbed_energy, "Coupling failed to transfer energy to Node B");

        // Reverse entire coupled system
        for &(u_a, u_b) in inputs.iter().rev() {
            lattice.step_backward(u_a, u_b);
        }

        println!("Coupled Lattice Inversion:");
        println!("  Initial A: ({}, {}) | Restored A: ({}, {})", 
            initial_state.0, initial_state.1, lattice.node_a.theta, lattice.node_a.p);
        println!("  Initial B: ({}, {}) | Restored B: ({}, {})", 
            initial_state.2, initial_state.3, lattice.node_b.theta, lattice.node_b.p);

        assert_eq!(lattice.node_a.theta, initial_state.0);
        assert_eq!(lattice.node_a.p, initial_state.1);
        assert_eq!(lattice.node_b.theta, initial_state.2);
        assert_eq!(lattice.node_b.p, initial_state.3);
    }
}
