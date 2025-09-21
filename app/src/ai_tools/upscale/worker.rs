use std::path::PathBuf;

use image::DynamicImage;
use serde_bytes::ByteBuf;
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::{
    ai_tools::{
        error::WorkerError,
        transport::{
            Transport,
            types::{LoadRequest, ModelKind, UpscaleRequest},
            zmq::ZmqTransport,
        },
    },
    config::UpscalingModel,
    worker::WorkerTrait,
};

pub struct Worker {
    transport: ZmqTransport,
    tx_image: Sender<DynamicImage>,
    rx_image: Receiver<DynamicImage>,
    tx_load: Sender<()>,
    rx_load: Receiver<()>,
    active_jobs: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl WorkerTrait for Worker {
    fn active_jobs(&self) -> std::sync::Arc<std::sync::Mutex<usize>> {
        self.active_jobs.clone()
    }
}

impl Worker {
    pub fn new() -> Self {
        let (tx_image, rx_image) = mpsc::channel(100);
        let (tx_load, rx_load) = mpsc::channel(100);

        Self {
            transport: ZmqTransport::new("tcp://127.0.0.1:5560"),
            tx_image,
            rx_image,
            tx_load,
            rx_load,
            active_jobs: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }

    pub fn upscale(&self, image: &DynamicImage, prompt: &str) {
        let image = image.clone();
        let prompt = prompt.to_owned();
        let client = self.transport.clone();

        self.run(self.tx_image.clone(), move || async move {
            let mut image_buf = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut image_buf),
                    image::ImageFormat::Png,
                )
                .map_err(WorkerError::ImageError)?;

            let response = client
                .send(UpscaleRequest {
                    prompt: prompt,
                    image_bytes: ByteBuf::from(image_buf),
                })
                .await
                .map_err(WorkerError::Transport)?;

            let inpainted_image =
                image::load_from_memory(&response.image).map_err(WorkerError::ImageError)?;

            Ok(inpainted_image)
        });
    }

    pub fn load(&self, kind: UpscalingModel, cache_dir: &PathBuf) {
        let client = self.transport.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();

        self.run(self.tx_load.clone(), move || async move {
            client
                .send(LoadRequest {
                    model: ModelKind::Upscaling(kind),
                    cache_dir,
                })
                .await
                .map_err(WorkerError::Transport)?;

            Ok(())
        });
    }

    pub fn upscaled(&mut self) -> Option<DynamicImage> {
        self.rx_image.try_recv().ok()
    }

    pub fn loaded(&mut self) -> bool {
        self.rx_load.try_recv().is_ok()
    }
}
