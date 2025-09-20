use std::path::PathBuf;
use tokio::sync::mpsc::{self, Receiver, Sender};

use image::{DynamicImage, GrayImage};
use serde_bytes::ByteBuf;

use crate::config::{ModelEntry, SelectionModel};
use crate::{
    ai_tools::transport::{
        Transport,
        types::{ImageSelectionRequest, LoadRequest, ModelKind},
        zmq::ZmqTransport,
    },
    worker::WorkerTrait,
};

pub struct Worker {
    client: ZmqTransport,
    tx_selections: Sender<Vec<GrayImage>>,
    rx_selections: Receiver<Vec<GrayImage>>,
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
        let (tx_image, rx_image) = mpsc::channel(100);
        let (tx_load, rx_load) = mpsc::channel(100);

        Self {
            client: ZmqTransport::new("tcp://127.0.0.1:5558"),
            tx_selections: tx_image,
            rx_selections: rx_image,
            tx_load,
            rx_load,
            active_jobs: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }

    pub fn image_selection(&self, image: &DynamicImage, prompt: &str, threshold: f32) {
        let image = image.clone();
        let prompt = prompt.to_owned();
        let client = self.client.clone();

        self.run(self.tx_selections.clone(), move || async move {
            let mut image_buf = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut image_buf),
                    image::ImageFormat::Png,
                )
                .unwrap();

            let response = client
                .send(ImageSelectionRequest {
                    prompt,
                    image_bytes: ByteBuf::from(image_buf),
                    threshold,
                })
                .await?;

            let selections = response
                .masks
                .into_iter()
                .filter_map(|mask_bytes| {
                    image::load_from_memory(&mask_bytes)
                        .ok()
                        .map(|img: DynamicImage| img.into_luma8())
                })
                .collect::<Vec<_>>();

            Ok(selections)
        });
    }

    pub fn load_model(&self, model: &ModelEntry<SelectionModel>, cache_dir: &PathBuf) {
        let client = self.client.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();
        let model = model.to_owned();

        self.run(self.tx_load.clone(), move || async move {
            client
                .send(LoadRequest {
                    model: ModelKind::Selection(model.kind),
                    cache_dir,
                })
                .await?;

            Ok(model.name)
        });
    }

    pub fn selections(&mut self) -> Option<Vec<GrayImage>> {
        self.rx_selections.try_recv().ok()
    }

    pub fn model_loaded(&mut self) -> Option<String> {
        self.rx_load.try_recv().ok()
    }
}
