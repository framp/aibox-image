use std::path::PathBuf;

use image::DynamicImage;
use serde_bytes::ByteBuf;
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::{
    ai_tools::{
        error::WorkerError,
        transport::{
            Transport,
            types::{FaceSwapRequest, LoadRequest, ModelKind},
            zmq::ZmqTransport,
        },
    },
    config::FaceSwappingModel,
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
            transport: ZmqTransport::new("tcp://127.0.0.1:5562"),
            tx_image,
            rx_image,
            tx_load,
            rx_load,
            active_jobs: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }

    pub fn swap_face(
        &self,
        source_image: &DynamicImage,
        target_image: &DynamicImage,
        face_index: i32,
    ) {
        let source_image = source_image.clone();
        let target_image = target_image.clone();
        let client = self.transport.clone();

        self.run(self.tx_image.clone(), move || async move {
            let mut source_buf = Vec::new();
            source_image
                .write_to(
                    &mut std::io::Cursor::new(&mut source_buf),
                    image::ImageFormat::Png,
                )
                .map_err(WorkerError::ImageError)?;

            let mut target_buf = Vec::new();
            target_image
                .write_to(
                    &mut std::io::Cursor::new(&mut target_buf),
                    image::ImageFormat::Png,
                )
                .map_err(WorkerError::ImageError)?;

            let response = client
                .send(FaceSwapRequest {
                    source_image_bytes: ByteBuf::from(source_buf),
                    target_image_bytes: ByteBuf::from(target_buf),
                    face_index,
                })
                .await
                .map_err(WorkerError::Transport)?;

            let swapped_image =
                image::load_from_memory(&response.image).map_err(WorkerError::ImageError)?;

            Ok(swapped_image)
        });
    }

    pub fn load(&self, kind: FaceSwappingModel, cache_dir: &PathBuf) {
        let client = self.transport.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();

        self.run(self.tx_load.clone(), move || async move {
            client
                .send(LoadRequest {
                    model: ModelKind::FaceSwapping(kind),
                    cache_dir,
                })
                .await
                .map_err(WorkerError::Transport)?;

            Ok(())
        });
    }

    pub fn swapped(&mut self) -> Option<DynamicImage> {
        self.rx_image.try_recv().ok()
    }

    pub fn loaded(&mut self) -> bool {
        self.rx_load.try_recv().is_ok()
    }
}
