use std::{
    collections::HashMap,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
};

use image::DynamicImage;
use serde_bytes::ByteBuf;

use crate::{
    ai_tools::transport::{
        Transport,
        types::{EditExpressionRequest, LoadRequest, ModelKind},
        zmq::ZmqTransport,
    },
    config::PortraitEditingModel,
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
        let (tx_image, rx_image) = mpsc::channel();
        let (tx_load, rx_load) = mpsc::channel();

        Self {
            transport: ZmqTransport::new("tcp://127.0.0.1:5561"),
            tx_image,
            rx_image,
            tx_load,
            rx_load,
            active_jobs: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }

    pub fn edit_expression(&self, image: &DynamicImage, expression_params: HashMap<String, f64>) {
        let image = image.clone();
        let client = self.transport.clone();

        self.run(self.tx_image.clone(), move || {
            let mut image_buf = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut image_buf),
                    image::ImageFormat::Png,
                )
                .unwrap();

            let response = client.send(EditExpressionRequest {
                image_bytes: ByteBuf::from(image_buf),
                expression_params,
            })?;

            let edited_image = image::load_from_memory(&response.image)
                .map_err(|e| format!("Failed to load image from response: {}", e))?;

            Ok(edited_image)
        });
    }

    pub fn load(&self, kind: PortraitEditingModel, cache_dir: &PathBuf) {
        let client = self.transport.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();

        self.run(self.tx_load.clone(), move || {
            client.send(LoadRequest {
                model: ModelKind::PortraitEditing(kind),
                cache_dir,
            })?;

            Ok(())
        });
    }

    pub fn edited_image(&self) -> Option<DynamicImage> {
        self.rx_image.try_recv().ok()
    }

    pub fn loaded(&self) -> bool {
        self.rx_load.try_recv().is_ok()
    }
}
