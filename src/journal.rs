use crate::coupled_lattice::LatticeEngine;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeJournalEntry {
    pub epoch: u64,
    pub q: Vec<f64>,
    pub p: Vec<f64>,
    pub energy: f64,
    pub dt: f64,
}

pub struct LatticeJournal {
    pub entries: Vec<LatticeJournalEntry>,
}

impl LatticeJournal {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn record_state(&mut self, lattice: &LatticeEngine, dt: f64) {
        self.entries.push(LatticeJournalEntry {
            epoch: lattice.epoch_count,
            q: lattice.q.clone(),
            p: lattice.p.clone(),
            energy: lattice.total_energy(),
            dt,
        });
    }

    pub fn replay_forward(&self, lattice: &mut LatticeEngine) {
        for entry in &self.entries {
            lattice.step_symplectic(entry.dt);
            lattice.update_phase_state();
        }
    }
}
