use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

#[derive(Debug, Clone)]
pub enum XrplStreamEvent {
    Tx { drops: u64, account: String },
    LedgerClosed {
        ledger_index: u64,
        close_time_resolution: u64,
        tx_count: u32,
    },
}

#[derive(Deserialize)]
struct LedgerClosedStreamMsg {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    ledger_index: Option<u64>,
    txn_count: Option<u32>,
    #[serde(rename = "close_time_resolution")]
    close_res: Option<u64>,
}

#[derive(Deserialize)]
struct TxStreamMsg {
    transaction: Option<TxInner>,
}

#[derive(Deserialize)]
struct TxInner {
    #[serde(rename = "TransactionType")]
    tx_type: Option<String>,
    #[serde(rename = "Account")]
    account: Option<String>,
    #[serde(rename = "Amount")]
    amount: Option<serde_json::Value>,
}

pub async fn start_xrpl_subscriber(tx: mpsc::Sender<XrplStreamEvent>) {
    let url = "wss://s1.ripple.com:51233";

    loop {
        if let Ok((ws_stream, _)) = connect_async(url).await {
            let (mut write, mut read) = ws_stream.split();

            let sub_cmd = serde_json::json!({
                "command": "subscribe",
                "streams": ["transactions", "ledger"]
            });

            if write.send(Message::Text(sub_cmd.to_string().into())).await.is_ok() {
                while let Some(Ok(msg)) = read.next().await {
                    if let Message::Text(text) = msg {
                        if let Ok(ledger) = serde_json::from_str::<LedgerClosedStreamMsg>(&text) {
                            if ledger.msg_type.as_deref() == Some("ledgerClosed") {
                                if let Some(idx) = ledger.ledger_index {
                                    let _ = tx.send(XrplStreamEvent::LedgerClosed {
                                        ledger_index: idx,
                                        close_time_resolution: ledger.close_res.unwrap_or(4),
                                        tx_count: ledger.txn_count.unwrap_or(0),
                                    }).await;
                                    continue;
                                }
                            }
                        }

                        if let Ok(tx_msg) = serde_json::from_str::<TxStreamMsg>(&text) {
                            if let Some(inner) = tx_msg.transaction {
                                if inner.tx_type.as_deref() == Some("Payment") {
                                    let drops = match inner.amount {
                                        Some(serde_json::Value::String(s)) => s.parse::<u64>().unwrap_or(1000),
                                        _ => 1000,
                                    };
                                    let acc = inner.account.unwrap_or_else(|| "unknown".into());
                                    let _ = tx.send(XrplStreamEvent::Tx { drops, account: acc }).await;
                                }
                            }
                        }
                    }
                }
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    }
}
