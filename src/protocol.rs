//! Wire-level message shapes for the alphanumeric pool's share-distribution
//! protocol (client side). Mirrors the pool backend's own `share_protocol.rs`
//! / `server.rs` module docs exactly:
//!
//! TCP, newline-delimited JSON, one JSON object per `\n`-terminated line.
//! Every line FROM the client is a request:
//! ```json
//! {"id": 1, "method": "subscribe", "params": null}
//! ```
//! Every line FROM the server in response to a request is exactly one of:
//! ```json
//! {"id": 1, "result": <value>}
//! {"id": 1, "error": "<message>"}
//! ```
//! Unsolicited server pushes have no `id`, and instead a `notify` field:
//! ```json
//! {"notify": "job", "params": <JobNotify>}
//! ```
//!
//! Methods: `subscribe` (no auth, returns a session id and triggers a `job`
//! push), `authorize` (params: [`AuthorizeParams`]), `submit` (params:
//! [`SubmitParams`]).
//!
//! ── PROVENANCE ──────────────────────────────────────────────────────────────
//! This module is COPIED VERBATIM from the alphanumeric CPU miner
//! (`alphanumeric-mining-pool-public/src/protocol.rs`). The GPU miner MUST speak
//! the byte-identical wire format, so this file is intentionally an exact copy;
//! keep it in sync with the CPU miner rather than diverging it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// alphanumeric addresses are exactly 40 lowercase hex chars
/// (`sha256(pubkey)[:20]`) -- no prefix, no checksum. Matches the pool
/// backend's own `is_valid_address` exactly: uppercase is rejected rather
/// than normalized, so an address that would round-trip differently than it
/// was typed is refused client-side before ever reaching the wire.
pub fn is_valid_address(address: &str) -> bool {
    address.len() == 40 && address.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

// ── Outgoing request params (client -> server) ──────────────────────────────

/// Params for `authorize`: identify a payout address + worker label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizeParams {
    pub address: String,
    pub worker: String,
}

/// Params for `submit`: a candidate nonce for `job_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitParams {
    pub job_id: String,
    pub nonce: u64,
}

/// A request envelope: `{"id", "method", "params"}`. Generic over the params
/// type so `subscribe` (params: null), `authorize`, and `submit` can all
/// share the same framing.
#[derive(Debug, Clone, Serialize)]
pub struct RequestEnvelope<T: Serialize> {
    pub id: u64,
    pub method: &'static str,
    pub params: T,
}

impl<T: Serialize> RequestEnvelope<T> {
    /// Render as a single `\n`-terminated JSON line, ready to write directly
    /// to the socket.
    pub fn to_line(&self) -> String {
        let mut line = serde_json::to_string(self).expect("request envelope always serializes");
        line.push('\n');
        line
    }
}

pub fn subscribe_request(id: u64) -> RequestEnvelope<Option<()>> {
    RequestEnvelope { id, method: "subscribe", params: None }
}

pub fn authorize_request(id: u64, address: String, worker: String) -> RequestEnvelope<AuthorizeParams> {
    RequestEnvelope { id, method: "authorize", params: AuthorizeParams { address, worker } }
}

pub fn submit_request(id: u64, job_id: String, nonce: u64) -> RequestEnvelope<SubmitParams> {
    RequestEnvelope { id, method: "submit", params: SubmitParams { job_id, nonce } }
}

// ── Incoming messages (server -> client) ────────────────────────────────────

/// Server -> client unsolicited job notification (the `params` of a `notify:
/// "job"` line). Field names/types mirror the pool backend's `JobNotify`
/// exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobNotify {
    pub job_id: String,
    pub number: u32,
    pub previous_hash: String, // hex
    pub merkle_root: String,   // hex
    pub timestamp: u64,
    pub difficulty: u64,
    pub target: String, // hex
}

/// A generic parse of any incoming line, before we know which of the three
/// shapes (`result`, `error`, `notify`) it is. Every field is optional so a
/// line missing/adding fields never fails to deserialize outright -- callers
/// inspect which fields are present to decide what the line means, matching
/// the server's own "never panic on a shape we don't expect" posture.
#[derive(Debug, Clone, Deserialize)]
pub struct IncomingLine {
    #[serde(default)]
    pub id: Value,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub notify: Option<String>,
    #[serde(default)]
    pub params: Option<Value>,
}

/// Failure decoding a [`JobNotify`]'s hex fields into fixed-size byte
/// arrays.
#[derive(Debug)]
pub enum JobDecodeError {
    BadHex { field: &'static str, source: hex::FromHexError },
    WrongLength { field: &'static str, got: usize },
}

impl std::fmt::Display for JobDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobDecodeError::BadHex { field, source } => write!(f, "field `{field}` is not valid hex: {source}"),
            JobDecodeError::WrongLength { field, got } => {
                write!(f, "field `{field}` decoded to {got} bytes, expected 32")
            }
        }
    }
}

impl std::error::Error for JobDecodeError {}

/// A [`JobNotify`] with its hex fields decoded into the fixed-size byte
/// arrays the PoW functions in [`crate::pow`] operate on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedJob {
    pub job_id: String,
    pub number: u32,
    pub previous_hash: [u8; 32],
    pub merkle_root: [u8; 32],
    pub timestamp: u64,
    pub difficulty: u64,
    /// The target this miner should grind against locally before
    /// submitting -- the pool's wire protocol only exposes the job's
    /// network target (not a separate, easier share target), so this is
    /// what a v1 client checks client-side.
    pub target: [u8; 32],
}

fn decode_hex_32(field: &'static str, hex_str: &str) -> Result<[u8; 32], JobDecodeError> {
    let bytes = hex::decode(hex_str).map_err(|source| JobDecodeError::BadHex { field, source })?;
    if bytes.len() != 32 {
        return Err(JobDecodeError::WrongLength { field, got: bytes.len() });
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

impl JobNotify {
    /// Decode this job's hex fields into fixed-size byte arrays. Returns an
    /// error (never panics) on malformed hex or wrong-length fields -- a
    /// pool sending a corrupt job should cause this job to be skipped, not
    /// the client to crash.
    pub fn decode(&self) -> Result<DecodedJob, JobDecodeError> {
        Ok(DecodedJob {
            job_id: self.job_id.clone(),
            number: self.number,
            previous_hash: decode_hex_32("previous_hash", &self.previous_hash)?,
            merkle_root: decode_hex_32("merkle_root", &self.merkle_root)?,
            timestamp: self.timestamp,
            difficulty: self.difficulty,
            target: decode_hex_32("target", &self.target)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_40_char_lowercase_hex_address_is_accepted() {
        assert!(is_valid_address(&"a".repeat(40)));
        assert!(is_valid_address("0123456789abcdef0123456789abcdef012345ab"));
    }

    #[test]
    fn wrong_length_address_is_rejected() {
        assert!(!is_valid_address(&"a".repeat(39)));
        assert!(!is_valid_address(&"a".repeat(41)));
        assert!(!is_valid_address(""));
    }

    #[test]
    fn uppercase_or_non_hex_address_is_rejected() {
        assert!(!is_valid_address(&"A".repeat(40)));
        assert!(!is_valid_address(&"g".repeat(40)));
        assert!(!is_valid_address(&"z".repeat(40)));
    }

    #[test]
    fn authorize_and_submit_params_round_trip_json() {
        let auth = AuthorizeParams { address: "a".repeat(40), worker: "rig1".into() };
        let json = serde_json::to_string(&auth).unwrap();
        let back: AuthorizeParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back, auth);

        let sub = SubmitParams { job_id: "j1".into(), nonce: 42 };
        let json = serde_json::to_string(&sub).unwrap();
        let back: SubmitParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back, sub);
    }

    #[test]
    fn job_notify_round_trips_json() {
        let job = JobNotify {
            job_id: "job-1".into(),
            number: 7,
            previous_hash: "ab".repeat(32),
            merkle_root: "cd".repeat(32),
            timestamp: 123,
            difficulty: 464,
            target: "ff".repeat(32),
        };
        let json = serde_json::to_string(&job).unwrap();
        let back: JobNotify = serde_json::from_str(&json).unwrap();
        assert_eq!(back, job);
    }

    #[test]
    fn subscribe_request_has_null_params() {
        let req = subscribe_request(1);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json, serde_json::json!({"id": 1, "method": "subscribe", "params": null}));
    }

    #[test]
    fn authorize_request_matches_server_expected_shape() {
        let req = authorize_request(2, "a".repeat(40), "rig1".to_string());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "id": 2,
                "method": "authorize",
                "params": {"address": "a".repeat(40), "worker": "rig1"}
            })
        );
    }

    #[test]
    fn submit_request_matches_server_expected_shape() {
        let req = submit_request(3, "job-1".to_string(), 42);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "id": 3,
                "method": "submit",
                "params": {"job_id": "job-1", "nonce": 42}
            })
        );
    }

    #[test]
    fn to_line_is_single_newline_terminated_json_object() {
        let req = subscribe_request(1);
        let line = req.to_line();
        assert!(line.ends_with('\n'));
        assert_eq!(line.matches('\n').count(), 1);
        // strip the trailing newline and confirm it parses back as the same object
        let parsed: Value = serde_json::from_str(line.trim_end()).unwrap();
        assert_eq!(parsed["method"], "subscribe");
    }

    #[test]
    fn incoming_line_parses_a_result_response() {
        let line = r#"{"id": 1, "result": {"session_id": "abc123"}}"#;
        let parsed: IncomingLine = serde_json::from_str(line).unwrap();
        assert_eq!(parsed.id, serde_json::json!(1));
        assert_eq!(parsed.result.unwrap()["session_id"], "abc123");
        assert!(parsed.error.is_none());
        assert!(parsed.notify.is_none());
    }

    #[test]
    fn incoming_line_parses_an_error_response() {
        let line = r#"{"id": 3, "error": "unknown job_id"}"#;
        let parsed: IncomingLine = serde_json::from_str(line).unwrap();
        assert_eq!(parsed.error.as_deref(), Some("unknown job_id"));
        assert!(parsed.result.is_none());
    }

    #[test]
    fn incoming_line_parses_a_job_notify_and_decodes_it() {
        let job = JobNotify {
            job_id: "job-1".into(),
            number: 7,
            previous_hash: "11".repeat(32),
            merkle_root: "22".repeat(32),
            timestamp: 1_000,
            difficulty: 464,
            target: "ff".repeat(32),
        };
        let line = serde_json::json!({"notify": "job", "params": job}).to_string();
        let parsed: IncomingLine = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed.notify.as_deref(), Some("job"));

        let notify: JobNotify = serde_json::from_value(parsed.params.unwrap()).unwrap();
        let decoded = notify.decode().unwrap();
        assert_eq!(decoded.job_id, "job-1");
        assert_eq!(decoded.previous_hash, [0x11u8; 32]);
        assert_eq!(decoded.merkle_root, [0x22u8; 32]);
        assert_eq!(decoded.target, [0xffu8; 32]);
    }

    #[test]
    fn decode_rejects_odd_length_hex() {
        let job = JobNotify {
            job_id: "job-1".into(),
            number: 1,
            previous_hash: "abc".into(), // odd number of hex digits
            merkle_root: "22".repeat(32),
            timestamp: 1,
            difficulty: 1,
            target: "ff".repeat(32),
        };
        assert!(matches!(job.decode(), Err(JobDecodeError::BadHex { field: "previous_hash", .. })));
    }

    #[test]
    fn decode_rejects_wrong_length_hex() {
        let job = JobNotify {
            job_id: "job-1".into(),
            number: 1,
            previous_hash: "ab".repeat(31), // valid hex, but only 31 bytes
            merkle_root: "22".repeat(32),
            timestamp: 1,
            difficulty: 1,
            target: "ff".repeat(32),
        };
        assert!(matches!(
            job.decode(),
            Err(JobDecodeError::WrongLength { field: "previous_hash", got: 31 })
        ));
    }

    #[test]
    fn incoming_line_never_panics_on_a_line_missing_every_field() {
        // `{}` is valid JSON and has none of id/result/error/notify/params --
        // must still parse (defaults kick in), never error out or panic.
        let parsed: IncomingLine = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.id, Value::Null);
        assert!(parsed.result.is_none());
    }
}
