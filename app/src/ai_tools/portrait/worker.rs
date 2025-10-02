use std::path::Path;

use image::DynamicImage;
use serde_bytes::ByteBuf;
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::{
    ai_tools::{
        error::AiToolError,
        transport::{
            Transport,
            types::{EditExpressionRequest, ExpressionParams, LoadRequest, ModelKind},
            zmq::ZmqTransport,
        },
    },
    config::PortraitEditingModel,
    worker::{ErrorChan, WorkerTrait},
};

pub struct Worker {
    transport: ZmqTransport,
    tx_image: Sender<DynamicImage>,
    rx_image: Receiver<DynamicImage>,
    tx_load: Sender<()>,
    rx_load: Receiver<()>,
    tx_err: ErrorChan,
    active_jobs: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl WorkerTrait for Worker {
    fn active_jobs(&self) -> std::sync::Arc<std::sync::Mutex<usize>> {
        self.active_jobs.clone()
    }
}

impl Worker {
    pub fn new(tx_err: ErrorChan) -> Self {
        let (tx_image, rx_image) = mpsc::channel(100);
        let (tx_load, rx_load) = mpsc::channel(100);

        Self {
            transport: ZmqTransport::new("tcp://127.0.0.1:5561"),
            tx_image,
            rx_image,
            tx_load,
            rx_load,
            tx_err,
            active_jobs: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }

    pub fn edit_expression(&self, image: &DynamicImage, expression_params: ExpressionParams) {
        let image = image.clone();
        let client = self.transport.clone();

        self.run(
            self.tx_image.clone(),
            self.tx_err.clone(),
            move || async move {
                let mut image_buf = Vec::new();
                image
                    .write_to(
                        &mut std::io::Cursor::new(&mut image_buf),
                        image::ImageFormat::Png,
                    )
                    .map_err(AiToolError::from)?;

                let response = client
                    .send(EditExpressionRequest {
                        image_bytes: ByteBuf::from(image_buf),
                        rotate_pitch: expression_params.rotate_pitch,
                        rotate_yaw: expression_params.rotate_yaw,
                        rotate_roll: expression_params.rotate_roll,
                        blink: expression_params.blink,
                        eyebrow: expression_params.eyebrow,
                        wink: expression_params.wink,
                        pupil_x: expression_params.pupil_x,
                        pupil_y: expression_params.pupil_y,
                        aaa: expression_params.aaa,
                        eee: expression_params.eee,
                        woo: expression_params.woo,
                        smile: expression_params.smile,
                        src_weight: expression_params.src_weight,
                    })
                    .await
                    .map_err(AiToolError::from)?;

                let edited_image =
                    image::load_from_memory(&response.image).map_err(AiToolError::from)?;

                Ok(edited_image)
            },
        );
    }

    pub fn load(&self, kind: PortraitEditingModel, cache_dir: &Path) {
        let client = self.transport.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();

        self.run(
            self.tx_load.clone(),
            self.tx_err.clone(),
            move || async move {
                client
                    .send(LoadRequest {
                        model: ModelKind::PortraitEditing(kind),
                        cache_dir,
                    })
                    .await
                    .map_err(AiToolError::from)?;

                Ok(())
            },
        );
    }

    pub fn edited_image(&mut self) -> Option<DynamicImage> {
        self.rx_image.try_recv().ok()
    }

    pub fn loaded(&mut self) -> bool {
        self.rx_load.try_recv().is_ok()
    }
}
