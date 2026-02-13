use anyhow::Context;
use futures::future::join_all;
use std::{path::Path, process::Command};
use tokio::process::Command as TokioCommand;

const UV_INSTALLER_URL: &str = "https://github.com/astral-sh/uv/releases/download/0.8.24";
const PYTHON_VERSION: &str = "3.11";

pub async fn run() -> Result<(), anyhow::Error> {
    install_uv()?;
    install_python()?;

    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().unwrap();
    let services_dir = exe_dir.join("services");

    let mut handles = vec![];

    for entry in std::fs::read_dir(services_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let handle = tokio::spawn(async move {
                uv_sync(&entry.path()).await?;
                uv_run_service(&entry.path()).await?;

                Ok::<(), anyhow::Error>(())
            });

            handles.push(handle);
        }
    }

    for handle in join_all(handles).await {
        handle??; // first ? unwraps JoinError, second ? unwraps anyhow::Error
    }

    Ok(())
}

fn install_uv() -> Result<(), anyhow::Error> {
    let mut cmd = Command::new("uv");
    cmd.arg("--help");

    let output = cmd
        .output()
        .with_context(|| "Failed to detect uv installation")?;

    if output.status.success() {
        return Ok(());
    }

    let mut cmd = {
        if cfg!(target_os = "windows") {
            let mut cmd = Command::new("powershell");
            cmd.args(&[
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                &format!("irm {UV_INSTALLER_URL}/uv-installer.ps1 | iex"),
            ]);
            cmd
        } else {
            let mut cmd = Command::new("sh");
            cmd.args(&[
                "-c",
                &format!(
                    "curl --proto '=https' --tlsv1.2 -LsSf {UV_INSTALLER_URL}/uv-installer.sh | sh"
                ),
            ]);
            cmd
        }
    };

    cmd.output().with_context(|| "Failed to install uv")?;

    Ok(())
}

fn install_python() -> Result<(), anyhow::Error> {
    let err = format!("Failed to run `uv install python {PYTHON_VERSION}`");

    let status = Command::new("uv")
        .args(&["python", "install", PYTHON_VERSION])
        .status()
        .with_context(|| err.clone())?;

    if !status.success() {
        anyhow::bail!(err);
    }

    Ok(())
}

async fn uv_sync(dir: &Path) -> Result<(), anyhow::Error> {
    let err = format!("Failed to run `uv sync` in {}", dir.display());

    let status = TokioCommand::new("uv")
        .arg("sync")
        .current_dir(dir)
        .status()
        .await
        .with_context(|| err.clone())?;

    if !status.success() {
        anyhow::bail!(err);
    }

    Ok(())
}

async fn uv_run_service(dir: &Path) -> Result<(), anyhow::Error> {
    let err = format!("Failed to run `uv run service` in {}", dir.display());

    let status = TokioCommand::new("uv")
        .args(&["run", "service"])
        .current_dir(dir)
        .status()
        .await
        .with_context(|| err.clone())?;

    if !status.success() {
        anyhow::bail!(err);
    }

    Ok(())
}
