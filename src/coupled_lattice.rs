use std::f64::consts::PI;

pub struct LatticeEngine {
    pub n_sites: usize,
    pub q: Vec<f64>,
    pub p: Vec<f64>,
    pub mass: f64,
    pub k_coupling: f64,
    pub d_damping: f64,
    pub e_max: f64,
    pub p_site_max: f64,
    pub phase_state: f64,
    pub epoch_count: u64,
}

impl LatticeEngine {
    pub fn new(n_sites: usize) -> Self {
        Self {
            n_sites,
            q: vec![0.0; n_sites],
            p: vec![0.0; n_sites],
            mass: 1.0,
            k_coupling: 0.8,
            d_damping: 0.05,
            e_max: 25000.0,
            p_site_max: 30000.0,
            phase_state: 0.0,
            epoch_count: 0,
        }
    }

    pub fn total_energy(&self) -> f64 {
        let mut kinetic = 0.0;
        let mut potential = 0.0;
        let n = self.n_sites;

        for i in 0..n {
            kinetic += 0.5 * self.p[i].powi(2) / self.mass;
            let next = (i + 1) % n;
            potential += 0.5 * self.k_coupling * (self.q[next] - self.q[i]).powi(2);
        }
        kinetic + potential
    }

    pub fn step_symplectic(&mut self, dt: f64) {
        let n = self.n_sites;
        let half_dt = 0.5 * dt;

        for i in 0..n {
            let left = if i == 0 { self.q[n - 1] } else { self.q[i - 1] };
            let right = if i == n - 1 { self.q[0] } else { self.q[i + 1] };
            let force = (right - 2.0 * self.q[i] + left) * self.k_coupling;
            let dissipation = self.d_damping * self.p[i];
            self.p[i] += half_dt * (force - dissipation);
        }

        for i in 0..n {
            self.q[i] += dt * (self.p[i] / self.mass);
        }

        for i in 0..n {
            let left = if i == 0 { self.q[n - 1] } else { self.q[i - 1] };
            let right = if i == n - 1 { self.q[0] } else { self.q[i + 1] };
            let force = (right - 2.0 * self.q[i] + left) * self.k_coupling;
            let dissipation = self.d_damping * self.p[i];
            self.p[i] += half_dt * (force - dissipation);
        }
    }

    pub fn update_phase_state(&mut self) {
        self.epoch_count += 1;
        let n = self.epoch_count as f64 + 1.0;
        let step = 1.5 * (n.ln() / n);
        self.phase_state = (self.phase_state + step) % (2.0 * PI);
    }

    pub fn evaluate_admission(&mut self, drops: u64, site_idx: usize) -> Result<(), String> {
        let impulse = (drops as f64).sqrt() * 0.01;
        let target_site = site_idx % self.n_sites;

        if self.p[target_site] + impulse > self.p_site_max {
            return Err(format!("P1 Flip: Q={}", drops));
        }

        let prospective_energy = self.total_energy() + impulse.powi(2) / (2.0 * self.mass);
        if prospective_energy > self.e_max {
            return Err(format!("E_max: {:.0}", prospective_energy));
        }

        self.p[target_site] += impulse;
        Ok(())
    }
}

// Alias to maintain compatibility with journal.rs
pub type CoupledLattice = LatticeEngine;
