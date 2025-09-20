use std::{
    io::Cursor,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
};

use image::{DynamicImage, GrayImage, Luma};
use serde_bytes::ByteBuf;

use crate::{
    ai_tools::transport::{
        Transport,
        types::{InpaintRequest, LoadRequest, ModelKind},
        zmq::ZmqTransport,
    },
    config::{InpaintingModel, ModelEntry},
    worker::WorkerTrait,
};

pub struct Worker {
    transport: ZmqTransport,
    tx_image: Sender<DynamicImage>,
    rx_image: Receiver<DynamicImage>,
    tx_load: Sender<String>,
    rx_load: Receiver<String>,
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
            transport: ZmqTransport::new("tcp://127.0.0.1:5559"),
            tx_image,
            rx_image,
            tx_load,
            rx_load,
            active_jobs: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }

    pub fn inpaint(&self, image: &DynamicImage, masks: Vec<GrayImage>, prompt: &str) {
        let image = image.clone();
        let prompt = prompt.to_owned();
        let client = self.transport.clone();

        self.run(self.tx_image.clone(), move || {
            let mask = merge_masks(&masks);
            let mut mask_buf = Vec::new();
            mask.write_to(&mut Cursor::new(&mut mask_buf), image::ImageFormat::Png)
                .unwrap();

            let mut image_buf = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut image_buf),
                    image::ImageFormat::Png,
                )
                .unwrap();

            let response = client.send(InpaintRequest {
                prompt: prompt,
                image_bytes: ByteBuf::from(image_buf),
                mask: ByteBuf::from(mask_buf),
            })?;

            let inpainted_image = image::load_from_memory(&response.image)
                .map_err(|e| format!("Failed to load image from response: {}", e))?;

            Ok(inpainted_image)
        });
    }

    pub fn load_model(&self, model: &ModelEntry<InpaintingModel>, cache_dir: &PathBuf) {
        let client = self.transport.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();
        let model = model.clone();

        self.run(self.tx_load.clone(), move || {
            client.send(LoadRequest {
                model: ModelKind::Inpainting(model.kind),
                cache_dir,
            })?;

            Ok(model.name)
        });
    }

    pub fn inpainted(&self) -> Option<DynamicImage> {
        self.rx_image.try_recv().ok()
    }

    pub fn model_loaded(&self) -> Option<String> {
        self.rx_load.try_recv().ok()
    }
}

fn merge_masks(images: &Vec<GrayImage>) -> GrayImage {
    if images.is_empty() {
        return GrayImage::new(1, 1);
    }

    let width = images.iter().map(|img| img.width()).max().unwrap();
    let height = images.iter().map(|img| img.height()).max().unwrap();

    let mut canvas: GrayImage = GrayImage::from_pixel(width, height, Luma([0]));

    for img in images {
        for (x, y, pixel) in img.enumerate_pixels() {
            if pixel[0] != 0 {
                canvas.put_pixel(x, y, *pixel);
            }
        }
    }

    canvas
}
