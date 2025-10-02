use std::{
    env,
    process::{Child, Command},
};

fn main() -> std::io::Result<()> {
    let exe_path = env::current_exe()?;
    let exe_dir = exe_path
        .parent()
        .expect("Failed to get executable directory");

    // Paths to binaries
    let main_app = exe_dir.join("./app/aibox-image");
    let workers = vec![
        exe_dir.join("./backend/face_swapping_service"),
        exe_dir.join("./backend/inpainting_service"),
        exe_dir.join("./backend/portrait_editing_service"),
        exe_dir.join("./backend/selection_service"),
        exe_dir.join("./backend/upscaling_service"),
    ];

    // Spawn all worker processes
    let mut worker_children: Vec<Child> = workers
        .into_iter()
        .map(|worker| {
            Command::new(worker.clone())
                .spawn()
                .expect(&format!("Failed to start {}", worker.display()))
        })
        .collect();

    // Spawn the main app
    let mut main_child = Command::new(main_app)
        .spawn()
        .expect("Failed to start aibox-image");

    // Wait for the main app to exit
    let status = main_child.wait()?;
    println!("aibox-image exited with status: {}", status);

    // Kill all worker processes
    for child in worker_children.iter_mut() {
        let _ = child.kill();
        let _ = child.wait();
    }

    println!("All workers terminated. Launcher exiting.");
    Ok(())
}
