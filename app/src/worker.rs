use std::{
    sync::{Arc, Mutex, mpsc::Sender},
    thread,
};

pub trait WorkerTrait {
    fn active_jobs(&self) -> Arc<Mutex<usize>>;

    fn is_processing(&self) -> bool {
        let jobs = self.active_jobs();
        let count = jobs.lock().unwrap();
        *count > 0
    }

    fn run<T, F>(&self, sender: Sender<T>, f: F)
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, Box<dyn std::error::Error>> + Send + 'static,
    {
        let jobs = self.active_jobs();

        thread::spawn(move || {
            {
                let mut count = jobs.lock().unwrap();
                *count += 1;
            }

            match f() {
                Ok(value) => {
                    let _ = sender.send(value);
                }
                Err(err) => {
                    eprintln!("Worker job failed: {err}");
                }
            }

            {
                let mut count = jobs.lock().unwrap();
                *count -= 1;
            }
        });
    }
}
