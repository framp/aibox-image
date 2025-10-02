use inflector::Inflector;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

fn main() {
    // Base folder
    let workspace_root = Path::new("..");

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    // Output folder for final executables
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let target_dir = PathBuf::from(&manifest_dir).join("target").join(&profile);

    let app_out_dir = target_dir.join("app");
    let backend_out_dir = target_dir.join("backend");

    fs::create_dir_all(&app_out_dir).expect("Failed to create app output dir");
    fs::create_dir_all(&backend_out_dir).expect("Failed to create backend output dir");

    let app_folder = workspace_root.join("app");
    build_rust_crate(&app_folder, &app_out_dir, &profile);
    println!("cargo:rerun-if-changed={}", app_folder.to_string_lossy());

    // Iterate over every app folder in backend
    for entry in fs::read_dir(workspace_root.join("backend")).expect("Failed to read backend dir") {
        let entry = entry.expect("Failed to read entry");
        let folder = entry.path();
        if !folder.is_dir() {
            continue;
        }

        build_backend_service(&folder, &backend_out_dir);

        println!("cargo:rerun-if-changed={}", folder.to_string_lossy());
    }
}

#[cfg(target_os = "windows")]
fn get_exe_name(base: &str) -> String {
    format!("{}.exe", base)
}

#[cfg(not(target_os = "windows"))]
fn get_exe_name(base: &str) -> String {
    base.to_string()
}

fn build_rust_crate(crate_dir: &Path, dist_dir: &Path, profile: &str) {
    let crate_name = read_crate_name(crate_dir);
    println!("cargo:warning=Building Rust crate: {}", crate_name);

    let args: &[&str] = match profile {
        "release" => &["build", "--release"],
        "debug" => &["build"],
        other => panic!("Unsupported Cargo profile: {}", other),
    };

    let status = Command::new("cargo")
        .args(args)
        .current_dir(crate_dir) // run from workspace root
        .status()
        .expect("Failed to run cargo build");

    if !status.success() {
        panic!("Failed to build Rust crate {}", crate_name);
    }

    let src_dir = crate_dir.join("target").join(profile);

    fs::copy(
        src_dir.join(&get_exe_name(&crate_name)),
        dist_dir.join(&get_exe_name(&crate_name)),
    )
    .expect("Failed to copy exe");

    fs::copy(src_dir.join("config.toml"), dist_dir.join("config.toml"))
        .expect("Failed to copy config.toml");

    println!(
        "cargo:warning=Copied {} to {}",
        src_dir.display(),
        dist_dir.display()
    );
}

fn build_backend_service(folder: &Path, dist_dir: &Path) {
    // Hooks directory
    let hooks_dir = Path::new(".");

    let app_name = folder
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_snake_case();

    // Construct src/<app_name>/main.py
    let abs_main_py = folder.join("src").join(&app_name).join("main.py");

    if !abs_main_py.exists() {
        println!("cargo:warning=Skipping {app_name} — main.py not found");
        return;
    }

    let main_py = abs_main_py
        .strip_prefix(&folder)
        .expect("Failed to get relative path to main.py from folder")
        .to_path_buf();

    let exe_name = &app_name;

    println!("cargo:warning=Building Python app with uv: {app_name}");

    // Run PyInstaller via uv
    let status = Command::new("uv")
        .env("PYTHONUNBUFFERED", "1")
        .current_dir(&folder)
        .args([
            "run",
            "pyinstaller",
            "--onefile",
            "--name",
            exe_name,
            "--additional-hooks-dir",
            hooks_dir.to_string_lossy().as_ref(),
            main_py.to_string_lossy().as_ref(),
        ])
        .status()
        .expect("failed to run PyInstaller");

    if !status.success() {
        panic!("Build failed for service: {}", app_name);
    }

    // Move the exe to launcher/dist
    let build_exe_name = get_exe_name(exe_name);
    let built_exe = folder.join("dist").join(&build_exe_name);
    let target_exe = dist_dir.join(&build_exe_name);

    fs::copy(&built_exe, &target_exe)
        .unwrap_or_else(|_| panic!("Failed to move {build_exe_name} to dist/"));

    println!(
        "cargo:warning=Copied {} to {}",
        built_exe.to_string_lossy(),
        target_exe.to_string_lossy()
    );
}

pub fn read_crate_name(crate_dir: &Path) -> String {
    let cargo_toml = crate_dir.join("Cargo.toml");
    let content =
        fs::read_to_string(&cargo_toml).expect(&format!("Failed to read {}", cargo_toml.display()));

    let value: toml::Value =
        toml::from_str(&content).expect(&format!("Failed to parse {}", cargo_toml.display()));

    value
        .get("package")
        .and_then(|pkg| pkg.get("name"))
        .and_then(|name| name.as_str())
        .expect("Missing [package] name in Cargo.toml")
        .to_string()
}
