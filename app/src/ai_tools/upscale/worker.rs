use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
};

use image::DynamicImage;
use serde_bytes::ByteBuf;

use crate::{
    ai_tools::transport::{
        Transport,
        types::{LoadRequest, ModelKind, UpscaleRequest},
        zmq::ZmqTransport,
    },
    config::UpscalingModel,
};

pub struct Worker {
    transport: ZmqTransport,
    tx_image: Sender<DynamicImage>,
    rx_image: Receiver<DynamicImage>,
    tx_load: Sender<()>,
    rx_load: Receiver<()>,
}

impl Worker {
    pub fn new() -> Self {
        let (tx_image, rx_image) = mpsc::channel();
        let (tx_load, rx_load) = mpsc::channel();

        Self {
            transport: ZmqTransport::new("tcp://127.0.0.1:5560"),
            tx_image,
            rx_image,
            tx_load,
            rx_load,
        }
    }

    pub fn upscale(&self, image: &DynamicImage, prompt: &str) {
        let image = image.clone();
        let prompt = prompt.to_owned();
        let client = self.transport.clone();

        crate::worker::run(self.tx_image.clone(), move || {
            let mut image_buf = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut image_buf),
                    image::ImageFormat::Png,
                )
                .unwrap();

            let response = client.send(UpscaleRequest {
                prompt: prompt,
                image_bytes: ByteBuf::from(image_buf),
            })?;

            let inpainted_image = image::load_from_memory(&response.image)
                .map_err(|e| format!("Failed to load image from response: {}", e))?;

            Ok(inpainted_image)
        });
    }

    pub fn load(&self, kind: UpscalingModel, cache_dir: &PathBuf) {
        let client = self.transport.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();

        crate::worker::run(self.tx_load.clone(), move || {
            client.send(LoadRequest {
                model: ModelKind::Upscaling(kind),
                cache_dir,
            })?;

            Ok(())
        });
    }

    pub fn upscaled(&self) -> Option<DynamicImage> {
        self.rx_image.try_recv().ok()
    }

    pub fn loaded(&self) -> bool {
        self.rx_load.try_recv().is_ok()
    }
}
