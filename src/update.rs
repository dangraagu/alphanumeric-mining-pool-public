//! SAFE, sha256-PINNED self-update (GPU miner).
//!
//! Mirrors the alphanumeric CPU miner's `src/update.rs` VERBATIM in logic; only
//! the manifest URL and the fallback binary name differ. See that module for the
//! full rationale. The security model is identical and summarised below.
//!
//! ## The whole security model is one sha256 gate
//! This module downloads and (opt-in) RUNS a binary. A blind download-and-run is
//! not acceptable. The entire trust model is a single check: the manifest
//! published at [`MANIFEST_URL`] carries the expected `sha256` of the new binary,
//! and [`apply_update`] installs the download ONLY if the sha256 of the bytes we
//! actually received equals that pinned value. That compare lives in exactly one
//! place -- [`verify_and_install`] -- and there is NO code path that installs a
//! binary without going through it. If the hash does not match, the downloaded
//! file is deleted and nothing is installed or executed.
//!
//! ## Failure is never fatal to mining
//! A failed check, download, sha256 mismatch, or install is LOGGED and the miner
//! keeps mining on its current version. An update problem must never stop a miner.
//!
//! ## Default is notify-only
//! With no flag the miner only CHECKS and LOGS. It installs nothing unless the
//! operator opted in with `--auto-update`. `--no-update-check` disables the check.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde::Deserialize;

/// Where the GPU miner fetches its version manifest from.
pub const MANIFEST_URL: &str = "https://alphanumeric.yamaduo.no/gpu-miner-version.json";

/// How often the reconnect loop re-checks for an update (~6 hours).
pub const CHECK_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

/// The JSON shape published at [`MANIFEST_URL`]:
/// ```json
/// { "version": "0.3.0",
///   "url": "https://alphanumeric.yamaduo.no/dl/alphanumeric-gpu-miner.exe",
///   "sha256": "<64-hex>" }
/// ```
/// All three fields are REQUIRED: a manifest missing `sha256` fails to parse, so
/// we can never reach the install path without a pinned hash.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct VersionManifest {
    pub version: String,
    pub url: String,
    pub sha256: String,
}

/// Result of a [`check_for_update`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateStatus {
    /// The published version is not newer than ours.
    UpToDate,
    /// A newer version is published. Carries what `apply_update` needs.
    Available { version: String, url: String, sha256: String },
    /// The check itself failed (network, bad JSON, ...). NEVER fatal -- the
    /// caller logs this and keeps mining.
    CheckFailed(String),
}

/// CLI-driven update behaviour, threaded into the reconnect loop.
#[derive(Debug, Clone, Copy)]
pub struct UpdateOptions {
    /// If false (`--no-update-check`), skip all update logic entirely.
    pub check_enabled: bool,
    /// If true (`--auto-update`), a found update is sha256-verified, installed,
    /// and re-exec'd. If false (the default), it is only logged.
    pub auto_update: bool,
}

impl Default for UpdateOptions {
    fn default() -> Self {
        // Safe default: check + notify, never auto-install.
        UpdateOptions { check_enabled: true, auto_update: false }
    }
}

// ── The check ────────────────────────────────────────────────────────────────

/// Fetch the manifest and compare its version to `current_version`. Never
/// panics; any failure becomes [`UpdateStatus::CheckFailed`] so the caller can
/// log and keep mining.
pub fn check_for_update(current_version: &str) -> UpdateStatus {
    let manifest = match fetch_manifest(MANIFEST_URL) {
        Ok(m) => m,
        Err(reason) => return UpdateStatus::CheckFailed(reason),
    };
    if is_newer(&manifest.version, current_version) {
        UpdateStatus::Available {
            version: manifest.version,
            url: manifest.url,
            sha256: manifest.sha256,
        }
    } else {
        UpdateStatus::UpToDate
    }
}

fn fetch_manifest(url: &str) -> Result<VersionManifest, String> {
    let body = agent()
        .get(url)
        .call()
        .map_err(|e| format!("could not fetch update manifest from {url}: {e}"))?
        .into_string()
        .map_err(|e| format!("could not read update manifest body: {e}"))?;
    serde_json::from_str::<VersionManifest>(&body)
        .map_err(|e| format!("update manifest is not valid JSON: {e}"))
}

/// SIMPLE semver compare (no semver crate): split on '.', compare each numeric
/// component. Returns true iff `candidate` is STRICTLY newer than `current`.
/// Missing trailing components count as 0, so "0.3" == "0.3.0". Non-numeric
/// components are treated as 0. Comparison is numeric, so "0.2.10" > "0.2.9".
pub fn is_newer(candidate: &str, current: &str) -> bool {
    let a = parse_version(candidate);
    let b = parse_version(current);
    let n = a.len().max(b.len());
    for i in 0..n {
        let ai = a.get(i).copied().unwrap_or(0);
        let bi = b.get(i).copied().unwrap_or(0);
        if ai != bi {
            return ai > bi;
        }
    }
    false // every component equal -> not newer
}

fn parse_version(v: &str) -> Vec<u64> {
    v.trim()
        .trim_start_matches('v')
        .split('.')
        // A component like "3" -> 3; anything non-numeric (or a "-rc1" suffix)
        // -> 0. Deliberately lenient: an unparseable manifest version simply
        // reads as older, it can never crash the miner.
        .map(|part| part.trim().parse::<u64>().unwrap_or(0))
        .collect()
}

// ── The apply (opt-in, sha256-VERIFIED) ──────────────────────────────────────

/// Download `url`, verify its sha256 against `expected_sha256`, and -- ONLY on a
/// match -- install it over the current executable. Returns the path of the
/// installed (new) binary on success.
///
/// SECURITY: the sha256 compare in [`verify_and_install`] gates every install.
/// A mismatch deletes the download and returns `Err`; nothing is ever installed
/// or executed unverified.
///
/// ## Windows rename-then-replace dance
/// On Windows a running `.exe` cannot be overwritten or deleted while it is
/// executing, but its directory entry CAN be renamed (the loaded image stays
/// mapped). So [`install_verified`] renames the running exe to `<exe>.old`, then
/// moves the verified download into the now-free path. The caller then re-execs
/// the new binary (see [`reexec`]) and exits; the next successful start removes
/// the leftover `<exe>.old` (see [`cleanup_old_binary`]).
pub fn apply_update(url: &str, expected_sha256: &str) -> Result<PathBuf, String> {
    let current_exe =
        std::env::current_exe().map_err(|e| format!("cannot locate the current executable: {e}"))?;
    let dir = current_exe
        .parent()
        .ok_or_else(|| "current executable has no parent directory".to_string())?;
    let tmp_path = dir.join(download_tmp_name(&current_exe));

    // Download the candidate binary to a temp file next to the current exe.
    let downloaded = download(url)?;
    std::fs::write(&tmp_path, &downloaded)
        .map_err(|e| format!("could not write download to {}: {e}", tmp_path.display()))?;

    // ─────────────────────────────────────────────────────────────────────────
    // MANDATORY sha256 GATE. Every install goes through this single function:
    // it hashes the downloaded bytes, compares to `expected_sha256`, installs
    // ONLY on a match, and deletes the temp file + errors on a mismatch. There
    // is deliberately no other path that installs a binary.
    // ─────────────────────────────────────────────────────────────────────────
    verify_and_install(&downloaded, expected_sha256, &tmp_path, &current_exe)?;

    Ok(current_exe)
}

/// THE SECURITY GATE. Verify the downloaded bytes against the pinned sha256 and
/// install ONLY on a match. On mismatch: delete the temp file and return `Err`
/// without touching the running binary. This is the one and only place an
/// install can happen, so the compare can never be accidentally skipped.
fn verify_and_install(
    downloaded: &[u8],
    expected_sha256: &str,
    tmp_path: &Path,
    current_exe: &Path,
) -> Result<(), String> {
    let actual = sha256_hex(downloaded);
    let expected = expected_sha256.trim();
    if !actual.eq_ignore_ascii_case(expected) {
        // Unverified binary: destroy it, install NOTHING, fail loudly.
        let _ = std::fs::remove_file(tmp_path);
        return Err(format!(
            "sha256 MISMATCH -- refusing to install. expected {expected}, downloaded {actual}. \
             Temp file deleted; the running binary was NOT touched."
        ));
    }
    // Verified. ONLY now do we install.
    install_verified(tmp_path, current_exe)
}

/// Move the running exe aside to `<exe>.old`, then move the (already sha256-
/// verified) temp file into the exe's path. Rolls back on failure so the miner
/// is never left with no binary at its path. Windows-specific rationale is in
/// [`apply_update`]'s doc.
fn install_verified(verified_tmp: &Path, current_exe: &Path) -> Result<(), String> {
    let old = old_path_for(current_exe);
    // Clear any stale `.old` from a previous update so the rename can't fail on
    // "file already exists".
    let _ = std::fs::remove_file(&old);
    // 1. Move the running exe out of the way (allowed while running; overwrite
    //    is not).
    std::fs::rename(current_exe, &old)
        .map_err(|e| format!("could not move current exe aside to {}: {e}", old.display()))?;
    // 2. Move the verified download into the now-free exe path.
    match std::fs::rename(verified_tmp, current_exe) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Roll back so the miner still has its exe where it was.
            let _ = std::fs::rename(&old, current_exe);
            Err(format!("could not install verified update into place: {e}"))
        }
    }
}

/// Re-exec the freshly-installed binary, forwarding our CLI args (skipping
/// argv[0]), then exit this old process. Never returns on success (the new
/// process takes over mining); returns `Err` only if the spawn itself fails, in
/// which case the caller keeps mining on the old, still-running image.
pub fn reexec(new_exe: &Path) -> Result<(), String> {
    let _child = Command::new(new_exe)
        .args(std::env::args().skip(1))
        .spawn()
        .map_err(|e| format!("update installed but could not launch new binary {}: {e}", new_exe.display()))?;
    // Detached hand-off: we do NOT wait. The child is the new miner; we exit so
    // it can (on Windows) later remove our now-`.old` image.
    std::process::exit(0);
}

/// Remove a leftover `<exe>.old` from a previous successful self-update, if
/// present. Best-effort; called once at startup and never fails the miner.
pub fn cleanup_old_binary() {
    if let Ok(exe) = std::env::current_exe() {
        let old = old_path_for(&exe);
        if old.exists() {
            match std::fs::remove_file(&old) {
                Ok(()) => println!("[update] cleaned up previous version at {}", old.display()),
                Err(e) => eprintln!(
                    "[update] note: could not remove old binary {} ({e}); harmless, continuing",
                    old.display()
                ),
            }
        }
    }
}

// ── Reconnect-loop hook ──────────────────────────────────────────────────────

/// One update tick, called from the reconnect loop's outer iteration. Throttled
/// to [`CHECK_INTERVAL`]: returns `last_check` unchanged when not due, otherwise
/// runs the check (and, under `--auto-update`, the verified apply + re-exec) and
/// returns the new check time. On a successful auto-apply this does NOT return
/// (the process re-execs). NEVER panics; any failure logs and mining continues.
pub fn tick(opts: UpdateOptions, last_check: Option<Instant>, current_version: &str) -> Option<Instant> {
    if !opts.check_enabled {
        return last_check;
    }
    let due = last_check.map_or(true, |t| t.elapsed() >= CHECK_INTERVAL);
    if !due {
        return last_check;
    }

    match check_for_update(current_version) {
        UpdateStatus::UpToDate => {
            println!("[update] up to date (v{current_version})");
        }
        UpdateStatus::CheckFailed(reason) => {
            // NEVER fatal -- keep mining on the current version.
            eprintln!("[update] check failed (harmless, still mining): {reason}");
        }
        UpdateStatus::Available { version, url, sha256 } => {
            if opts.auto_update {
                println!("[update] v{version} available -- auto-updating (sha256-verified)...");
                match apply_update(&url, &sha256) {
                    Ok(new_exe) => {
                        println!(
                            "[update] verified + installed v{version}; restarting into {}",
                            new_exe.display()
                        );
                        // Never returns on success (process re-execs).
                        if let Err(e) = reexec(&new_exe) {
                            eprintln!("[update] installed but re-exec failed (still mining): {e}");
                        }
                    }
                    Err(e) => {
                        // sha256 mismatch, download error, or install error --
                        // LOUD, but mining continues on the current version.
                        eprintln!(
                            "[update] AUTO-UPDATE FAILED -- staying on v{current_version} and \
                             continuing to mine: {e}"
                        );
                    }
                }
            } else {
                println!(
                    "[update] v{version} available -- run with --auto-update to apply, \
                     or download from {url}"
                );
            }
        }
    }
    Some(Instant::now())
}

// ── Small helpers ────────────────────────────────────────────────────────────

/// A blocking HTTP agent with bounded timeouts so a hung server can never wedge
/// the miner. rustls-backed (ureq default features) so https works with no
/// system OpenSSL on Windows.
fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(60))
        .build()
}

fn download(url: &str) -> Result<Vec<u8>, String> {
    let resp = agent()
        .get(url)
        .call()
        .map_err(|e| format!("download failed from {url}: {e}"))?;
    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("could not read downloaded bytes from {url}: {e}"))?;
    Ok(bytes)
}

/// sha256 of `bytes` as lowercase hex. The core of the security gate; unit-
/// tested against known vectors.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn download_tmp_name(current_exe: &Path) -> String {
    let base = current_exe
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("alphanumeric-gpu-miner");
    format!("{base}.update-download")
}

/// `<exe>` -> `<exe>.old` (e.g. `alphanumeric-gpu-miner.exe.old`). Appends to
/// the whole file name rather than replacing the extension, so the `.old` marker
/// is unambiguous and matches what [`cleanup_old_binary`] looks for.
fn old_path_for(exe: &Path) -> PathBuf {
    let mut s = exe.as_os_str().to_owned();
    s.push(".old");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── semver compare ───────────────────────────────────────────────────────

    #[test]
    fn is_newer_detects_a_newer_version() {
        assert!(is_newer("0.3.0", "0.2.9"));
        assert!(is_newer("1.0.0", "0.9.9"));
        assert!(is_newer("0.3.1", "0.3.0"));
        // numeric, NOT lexical: 10 > 9
        assert!(is_newer("0.2.10", "0.2.9"));
    }

    #[test]
    fn is_newer_is_false_for_equal_versions() {
        assert!(!is_newer("0.3.0", "0.3.0"));
        assert!(!is_newer("1.2.3", "1.2.3"));
    }

    #[test]
    fn is_newer_is_false_for_older_versions() {
        assert!(!is_newer("0.2.9", "0.3.0"));
        assert!(!is_newer("1.0.0", "2.0.0"));
        // numeric, NOT lexical: 9 < 10
        assert!(!is_newer("0.2.9", "0.2.10"));
    }

    #[test]
    fn is_newer_handles_different_component_counts() {
        // Missing trailing components are treated as 0.
        assert!(!is_newer("0.3", "0.3.0")); // equal
        assert!(!is_newer("0.3.0", "0.3")); // equal
        assert!(is_newer("0.3.1", "0.3")); // 0.3.1 > 0.3.0
        assert!(is_newer("1", "0.9.9")); // 1.0.0 > 0.9.9
        assert!(!is_newer("0.3", "0.3.1")); // 0.3.0 < 0.3.1
    }

    #[test]
    fn is_newer_tolerates_v_prefix_and_junk() {
        assert!(is_newer("v0.3.0", "0.2.9"));
        assert!(!is_newer("garbage", "0.1.0")); // parses as 0 -> not newer
    }

    // ── manifest parse ───────────────────────────────────────────────────────

    #[test]
    fn manifest_parses_expected_json_shape() {
        let json = r#"{
            "version": "0.3.0",
            "url": "https://alphanumeric.yamaduo.no/dl/alphanumeric-gpu-miner.exe",
            "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        }"#;
        let m: VersionManifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.version, "0.3.0");
        assert_eq!(m.url, "https://alphanumeric.yamaduo.no/dl/alphanumeric-gpu-miner.exe");
        assert_eq!(m.sha256.len(), 64);
    }

    #[test]
    fn manifest_parse_rejects_missing_sha256() {
        // Missing sha256 MUST fail: we can never reach install without a hash.
        let json = r#"{"version":"0.3.0","url":"https://x/y.exe"}"#;
        assert!(serde_json::from_str::<VersionManifest>(json).is_err());
    }

    // ── the sha256 gate itself ───────────────────────────────────────────────

    #[test]
    fn sha256_hex_matches_known_vectors() {
        // sha256("") and sha256("abc"), the canonical NIST vectors.
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn verify_and_install_refuses_and_deletes_on_sha_mismatch() {
        // Exercises the gate directly (no network, no re-exec): a wrong expected
        // hash must delete the temp file, error, and NEVER touch current_exe.
        let dir = std::env::temp_dir();
        let tmp = dir.join(format!("alphanumeric-gpu-update-test-{}.tmp", std::process::id()));
        std::fs::write(&tmp, b"not the real binary").unwrap();
        let fake_current = dir.join(format!("alphanumeric-gpu-update-never-touch-{}", std::process::id()));

        let wrong_expected = "0".repeat(64);
        let res = verify_and_install(b"not the real binary", &wrong_expected, &tmp, &fake_current);

        assert!(res.is_err(), "a sha mismatch must return Err");
        assert!(!tmp.exists(), "the unverified download must be deleted on mismatch");
        assert!(!fake_current.exists(), "the current exe path must never be created on mismatch");
        let _ = std::fs::remove_file(&tmp); // cleanup if the assert path left it
    }
}
