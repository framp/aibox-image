use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
};

use image::{DynamicImage, GrayImage};
use serde_bytes::ByteBuf;

use crate::ai_tools::transport::{
    Transport,
    types::{ImageSelectionRequest, LoadRequest, ModelKind},
    zmq::ZmqTransport,
};
use crate::config::SelectionModel;

pub struct Worker {
    client: ZmqTransport,
    tx_selections: Sender<Vec<GrayImage>>,
    rx_selections: Receiver<Vec<GrayImage>>,
    tx_load: Sender<()>,
    rx_load: Receiver<()>,
}

impl Worker {
    pub fn new() -> Self {
        let (tx_image, rx_image) = mpsc::channel();
        let (tx_load, rx_load) = mpsc::channel();

        Self {
            client: ZmqTransport::new("tcp://127.0.0.1:5558"),
            tx_selections: tx_image,
            rx_selections: rx_image,
            tx_load,
            rx_load,
        }
    }

    pub fn image_selection(&self, image: &DynamicImage, prompt: &str, threshold: f32) {
        let image = image.clone();
        let prompt = prompt.to_owned();
        let client = self.client.clone();

        crate::worker::run(self.tx_selections.clone(), move || {
            let mut image_buf = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut image_buf),
                    image::ImageFormat::Png,
                )
                .unwrap();

            let response = client.send(ImageSelectionRequest {
                prompt,
                image_bytes: ByteBuf::from(image_buf),
                threshold,
            })?;

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

    pub fn load(&self, kind: SelectionModel, cache_dir: &PathBuf) {
        let client = self.client.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();

        crate::worker::run(self.tx_load.clone(), move || {
            client.send(LoadRequest {
                model: ModelKind::Selection(kind),
                cache_dir,
            })?;

            Ok(())
        });
    }

    pub fn selected(&self) -> Option<Vec<GrayImage>> {
        self.rx_selections.try_recv().ok()
    }

    pub fn loaded(&self) -> bool {
        self.rx_load.try_recv().is_ok()
    }
}
