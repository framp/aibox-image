use std::{sync::mpsc::Sender, thread};

pub fn run<T, F>(sender: Sender<T>, f: F)
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, Box<dyn std::error::Error>> + Send + 'static,
{
    thread::spawn(move || {
        let result = f().unwrap();
        sender.send(result).unwrap();
    });
}
