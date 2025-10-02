use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::Sender;

use crate::error::Error;

pub type ErrorChan = Sender<Error>;

pub trait WorkerTrait {
    fn active_jobs(&self) -> Arc<Mutex<usize>>;

    fn is_processing(&self) -> bool {
        let jobs = self.active_jobs();
        let count = jobs.lock().unwrap();
        *count > 0
    }

    fn run<T, F, Fut>(&self, sender: Sender<T>, error: ErrorChan, f: F)
    where
        T: Send + 'static,
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T, Error>> + Send + 'static,
    {
        let jobs = self.active_jobs();

        tokio::spawn(async move {
            {
                let mut count = jobs.lock().unwrap();
                *count += 1;
            }

            match f().await {
                Ok(value) => {
                    let _ = sender.send(value).await;
                }
                Err(err) => {
                    let _ = error.send(err).await;
                }
            }

            {
                let mut count = jobs.lock().unwrap();
                *count -= 1;
            }
        });
    }
}
