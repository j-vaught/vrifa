use anyhow::{bail, Context, Result};
use fast_vrifa_core::{DelegatedCpuBackend, ImageBackend};
use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

pub use vrifa_cli::Config;

pub fn run() -> Result<()> {
    let status = forward_to_reference(env::args_os().skip(1))?;
    if status.success() {
        return Ok(());
    }
    if let Some(code) = status.code() {
        bail!("reference vrifa exited with status code {code}");
    }
    bail!("reference vrifa terminated by signal");
}

pub fn run_config(config: Config) -> Result<()> {
    vrifa_cli::run_binding_config(config).context("delegating bound config to reference vrifa")
}

pub fn delegated_backend_label() -> &'static str {
    let backend = DelegatedCpuBackend;
    backend.label()
}

pub fn reference_binary_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("VRIFA_BIN") {
        candidates.push(PathBuf::from(path));
    }

    let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_root.join("../../..");
    candidates.push(repo_root.join("vrifa-rs/target/release/vrifa"));
    candidates.push(repo_root.join("vrifa-rs/target/debug/vrifa"));
    candidates.push(PathBuf::from("vrifa"));
    candidates
}

pub fn locate_reference_binary() -> Result<PathBuf> {
    let candidates = reference_binary_candidates();
    for candidate in &candidates {
        if candidate.is_file() || candidate == Path::new("vrifa") {
            return Ok(candidate.clone());
        }
    }
    bail!("unable to locate the locked vrifa binary; build vrifa-rs first or set VRIFA_BIN")
}

pub fn forward_to_reference<I, S>(args: I) -> Result<ExitStatus>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let reference = locate_reference_binary()?;
    Command::new(&reference)
        .args(args)
        .status()
        .with_context(|| format!("launching delegated binary {}", reference.display()))
}

#[cfg(test)]
mod tests {
    use super::{delegated_backend_label, reference_binary_candidates};

    #[test]
    fn delegated_backend_is_reported() {
        assert_eq!(delegated_backend_label(), "delegated-cpu");
    }

    #[test]
    fn reference_binary_search_order_is_seeded() {
        let candidates = reference_binary_candidates();
        assert!(candidates.len() >= 3);
    }
}
