use crate::coupled_lattice::CoupledLattice;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatticeState {
    pub theta_a: u32,
    pub p_a: i64,
    pub theta_b: u32,
    pub p_b: i64,
}

impl LatticeState {
    pub fn capture(lattice: &CoupledLattice) -> Self {
        Self {
            theta_a: lattice.node_a.theta,
            p_a: lattice.node_a.p,
            theta_b: lattice.node_b.theta,
            p_b: lattice.node_b.p,
        }
    }
}

pub struct ReversibleJournal {
    capacity: usize,
    buffer: Vec<(i64, i64)>,
    head: usize,
    count: usize,
}

impl ReversibleJournal {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: vec![(0, 0); capacity],
            head: 0,
            count: 0,
        }
    }

    /// Records an input pair and steps the lattice forward
    #[inline(always)]
    pub fn step(&mut self, lattice: &mut CoupledLattice, u_a: i64, u_b: i64) {
        lattice.step_forward(u_a, u_b);

        self.buffer[self.head] = (u_a, u_b);
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Steps the lattice backward by a specified number of logged steps
    pub fn rewind(&mut self, lattice: &mut CoupledLattice, steps_to_rewind: usize) -> Result<usize, &'static str> {
        let actual_steps = steps_to_rewind.min(self.count);
        if actual_steps == 0 {
            return Ok(0);
        }

        for _ in 0..actual_steps {
            self.head = (self.head + self.capacity - 1) % self.capacity;
            let (u_a, u_b) = self.buffer[self.head];
            lattice.step_backward(u_a, u_b);
            self.count -= 1;
        }

        Ok(actual_steps)
    }

    pub fn available_history(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_journal_bounded_rewind() {
        let mut lattice = CoupledLattice::new(0, 0, 40);
        let buffer_size = 50_000;
        let mut journal = ReversibleJournal::new(buffer_size);

        // Run warm-up steps
        for step in 0..10_000 {
            journal.step(&mut lattice, (step % 17) - 8, (step % 23) - 11);
        }

        // Snapshot checkpoint at step 10,000
        let checkpoint = LatticeState::capture(&lattice);

        // Advance further by 25,000 steps
        for step in 10_000..35_000 {
            let u_a = (((step % 101) as i64) - 50) * 3;
            let u_b = (((step % 79) as i64) - 39) * 3;
            journal.step(&mut lattice, u_a, u_b);
        }

        let state_after_run = LatticeState::capture(&lattice);
        assert_ne!(state_after_run, checkpoint, "Lattice state did not evolve");

        // Rewind 25,000 steps back to the checkpoint
        let rewound = journal.rewind(&mut lattice, 25_000).unwrap();
        assert_eq!(rewound, 25_000);

        let restored_state = LatticeState::capture(&lattice);
        println!("Lattice Checkpoint Inversion:");
        println!("  Checkpoint: {:?}", checkpoint);
        println!("  Restored:   {:?}", restored_state);

        assert_eq!(restored_state, checkpoint, "Failed to restore exact checkpoint");
    }
}
