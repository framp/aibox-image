use std::fs;
use std::path::PathBuf;

fn main() {
    let profile = std::env::var("PROFILE").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let target_dir = PathBuf::from(&manifest_dir).join("target").join(&profile);

    let src = PathBuf::from(&manifest_dir).join("config.toml");
    let dst = target_dir.join("config.toml");

    fs::copy(&src, &dst).expect("Failed to copy config.toml");
    println!("cargo:rerun-if-changed=config.toml");
}
