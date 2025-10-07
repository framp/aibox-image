use fs_extra::dir;
use std::fs;
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    let profile = std::env::var("PROFILE").unwrap();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let manifest_dir = PathBuf::from(manifest_dir);
    let target_dir = PathBuf::from(&manifest_dir).join("target").join(&profile);

    copy_config(&manifest_dir, &target_dir).expect("Failed to copy config.toml");

    let backend_dir = PathBuf::from(&manifest_dir).join("../backend");
    for entry in fs::read_dir(&backend_dir).expect("Failed to read backend directory") {
        let entry = entry.expect("Failed to read entry");

        if entry.file_type().expect("Failed to get file type").is_dir() {
            let service_dir = backend_dir.join(entry.file_name());
            let target_service_dir = target_dir.join("services").join(entry.file_name());

            copy_python_project(&service_dir, &target_service_dir)
                .expect(&format!("Failed to copy {}", service_dir.display()));
        }
    }
}

fn copy_config(manifest_dir: &PathBuf, target_dir: &PathBuf) -> Result<()> {
    let src = PathBuf::from(&manifest_dir).join("config.toml");
    let dst = target_dir.join("config.toml");

    fs::copy(&src, &dst)?;
    println!("cargo:rerun-if-changed=config.toml");

    Ok(())
}

fn copy_python_project(src_dir: &PathBuf, target_dir: &PathBuf) -> Result<()> {
    fs::create_dir_all(&target_dir)?;

    fs::copy(
        &src_dir.join("pyproject.toml"),
        &target_dir.join("pyproject.toml"),
    )?;

    fs::copy(&src_dir.join("uv.lock"), &target_dir.join("uv.lock"))?;

    dir::copy(
        &src_dir.join("src"),
        &target_dir,
        &dir::CopyOptions::default()
            .overwrite(true)
            .copy_inside(true),
    )?;

    // TODO: hack
    let _ = dir::copy(
        &src_dir.join("LivePortrait"),
        &target_dir,
        &dir::CopyOptions::default()
            .overwrite(true)
            .copy_inside(true),
    );

    println!("cargo:rerun-if-changed={}", src_dir.display());
    Ok(())
}
