use serde::Deserialize;
use crate::{SymplecticDDS, TelemetryCoupler};

pub const RIPPLE_EPOCH_OFFSET: i64 = 946_684_800;

#[derive(Debug, Deserialize)]
pub struct XrplRpcResponse<T> {
    pub result: T,
    pub status: String,
}

#[derive(Debug, Deserialize)]
pub struct TxResult {
    #[serde(default)]
    pub validated: bool,
    #[serde(rename = "Fee")]
    pub fee: Option<String>,
    #[serde(rename = "Amount")]
    pub amount: Option<serde_json::Value>,
    pub date: Option<i64>,
    #[serde(rename = "ledger_index")]
    pub ledger_index: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct LedgerHeader {
    pub close_time: i64,
    pub close_time_resolution: u32,
    pub total_coins: String,
    pub ledger_index: u64,
}

#[derive(Debug, Deserialize)]
pub struct LedgerResult {
    pub ledger: Option<LedgerHeader>,
    pub validated: bool,
}

pub struct XrplTelemetryAdapter {
    pub coupler: TelemetryCoupler,
    last_close_time: Option<i64>,
    drops_normalization: f64,
}

impl XrplTelemetryAdapter {
    pub fn new(coupler: TelemetryCoupler, drops_normalization: f64) -> Self {
        Self {
            coupler,
            last_close_time: None,
            drops_normalization, // e.g., 1e-6 to map drops -> XRP, or 1e-8 for scaling
        }
    }

    /// Extracts raw XRP drops from an ambiguous Amount field (native drops or issued currency)
    pub fn extract_drops(amount_val: &serde_json::Value) -> Option<u64> {
        match amount_val {
            // Native XRP arrives as a string containing drops integer
            serde_json::Value::String(s) => s.parse::<u64>().ok(),
            // Non-XRP issued tokens arrive as {"currency": "...", "value": "..."}
            serde_json::Value::Object(map) => {
                map.get("value")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|v| (v * 1_000_000.0) as u64)
            }
            _ => None,
        }
    }

    /// Converts a raw JSON-RPC response from `tx` into an energy-bounded input_shift
    pub fn parse_tx_to_input(
        &mut self,
        json_str: &str,
        sim: &SymplecticDDS,
    ) -> Result<i64, String> {
        let resp: XrplRpcResponse<TxResult> =
            serde_json::from_str(json_str).map_err(|e| format!("JSON Parse Error: {}", e))?;

        if resp.status != "success" && !resp.result.validated {
            return Err("Transaction not validated or query failed".to_string());
        }

        let drops = resp
            .result
            .amount
            .as_ref()
            .and_then(Self::extract_drops)
            .unwrap_or(0);

        // Compute time-cadence weight if ledger close timestamp is present
        let dt_factor = if let Some(close_date) = resp.result.date {
            let prev = self.last_close_time.unwrap_or(close_date);
            self.last_close_time = Some(close_date);
            let dt = (close_date - prev).max(1);
            1.0 / (dt as f64)
        } else {
            1.0
        };

        // Scalar drive signal: (Drops * Cadence) normalized to float
        let raw_signal = (drops as f64 * self.drops_normalization) * dt_factor;

        // Condition through separatrix envelope
        Ok(self.coupler.condition_input(raw_signal, sim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_xrpl_tx_payload() {
        let sample_rpc_response = r#"{
            "result": {
                "Account": "rUFCLJpWcd2yng9tvmCfV7Pq5yG59XTbsX",
                "Amount": "100941437",
                "Destination": "r9N4GDbG3vPujNAW9KYxWAvDUaZmEr3BRu",
                "Fee": "12",
                "date": 745812930,
                "ledger_index": 106750290,
                "validated": true
            },
            "status": "success"
        }"#;

        let sim = SymplecticDDS::new(0);
        let coupler = TelemetryCoupler::new(10, 500_000);
        // Normalize 100 XRP drops to ~100.0 scalar
        let mut adapter = XrplTelemetryAdapter::new(coupler, 1e-6);

        let input_shift = adapter.parse_tx_to_input(sample_rpc_response, &sim);
        assert!(input_shift.is_ok());

        let u = input_shift.unwrap();
        // 100.941437 * 10 * 1.0 ≈ 1009
        assert!(u > 1000 && u < 1020, "Conditioned input mismatch: got {}", u);
    }
}
