//! GF(3) Symplectic Parity Check & Attestation Engine for [[9,1,3]]_3

pub const NUM_QUTRITS: usize = 9;
pub const NUM_GENERATORS: usize = 8;
pub const SYMPLECTIC_DIM: usize = 18;

pub struct GF3StabilizerEngine {
    u_matrix: [[u8; NUM_QUTRITS]; NUM_GENERATORS],
    v_matrix: [[u8; NUM_QUTRITS]; NUM_GENERATORS],
}

impl Default for GF3StabilizerEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GF3StabilizerEngine {
    pub fn new() -> Self {
        let mut u = [[0u8; NUM_QUTRITS]; NUM_GENERATORS];
        let mut v = [[0u8; NUM_QUTRITS]; NUM_GENERATORS];

        // Rows 0-1: Block 1 Z-checks (Qutrits 0, 1, 2)
        v[0][0] = 1; v[0][1] = 2;
        v[1][1] = 1; v[1][2] = 2;

        // Rows 2-3: Block 2 Z-checks (Qutrits 3, 4, 5)
        v[2][3] = 1; v[2][4] = 2;
        v[3][4] = 1; v[3][5] = 2;

        // Rows 4-5: Block 3 Z-checks (Qutrits 6, 7, 8)
        v[4][6] = 1; v[4][7] = 2;
        v[5][7] = 1; v[5][8] = 2;

        // Rows 6-7: X-type inter-block checks
        u[6] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
        u[7] = [0, 0, 0, 2, 2, 2, 1, 1, 1];

        Self {
            u_matrix: u,
            v_matrix: v,
        }
    }

    /// Extract 8-dimensional ternary syndrome over GF(3): (u . v_err - v . u_err) mod 3
    pub fn extract_syndrome(&self, error_vector: &[u8; SYMPLECTIC_DIM]) -> [u8; NUM_GENERATORS] {
        let u_err = &error_vector[..NUM_QUTRITS];
        let v_err = &error_vector[NUM_QUTRITS..];
        let mut syndrome = [0u8; NUM_GENERATORS];

        for r in 0..NUM_GENERATORS {
            let mut acc: i32 = 0;
            for i in 0..NUM_QUTRITS {
                acc += (self.u_matrix[r][i] as i32) * (v_err[i] as i32);
                acc -= (self.v_matrix[r][i] as i32) * (u_err[i] as i32);
            }
            let rem = acc % 3;
            syndrome[r] = if rem < 0 { (rem + 3) as u8 } else { rem as u8 };
        }
        syndrome
    }

    /// Encode payload into hexadecimal XRPL Payment Memo format
    pub fn format_xrpl_memo_json(
        &self,
        sequence: u64,
        root_tx: &str,
        verifier: &str,
        syndrome: &[u8; NUM_GENERATORS],
        x_coord: f64,
        y_coord: f64,
    ) -> String {
        format!(
            "{{\"seq\":{},\"root\":\"{}\",\"verifier\":\"{}\",\"syndrome\":{:?},\"coords\":{{\"x\":{:.6},\"y\":{:.6}}}}}",
            sequence, root_tx, verifier, syndrome, x_coord, y_coord
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_flip_syndrome() {
        let engine = GF3StabilizerEngine::new();
        // Z error on Qutrit #4 -> v_4 = 1
        let mut err = [0u8; SYMPLECTIC_DIM];
        err[9 + 4] = 1;

        let syn = engine.extract_syndrome(&err);
        assert_eq!(syn, [0, 0, 0, 0, 0, 0, 1, 2]);
    }
}
