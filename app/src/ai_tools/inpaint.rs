use std::{
    io::Cursor,
    sync::mpsc::{self, Receiver, Sender},
};

use eframe::egui::{Button, TextEdit, Ui};
use image::{DynamicImage, GrayImage, Luma};
use serde_bytes::ByteBuf;

use crate::{
    ai_tools::zmq::InpaintPayload, image_canvas::ImageCanvas, layer_system::LayerOperation,
    mask_gallery::MaskGallery, undo_redo::UndoRedoManager,
};

use super::zmq;

pub struct InpaintTool {
    input: String,
    loading: bool,
    tx: Sender<DynamicImage>,
    rx: Receiver<DynamicImage>,
}

impl InpaintTool {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();

        Self {
            input: String::new(),
            loading: false,
            tx,
            rx,
        }
    }

    fn fetch(&mut self, canvas: &ImageCanvas) {
        self.loading = true;

        let image_path = canvas
            .image_path
            .as_ref()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let masks = canvas.selections.iter().map(|s| s.mask.clone()).collect();
        let input = self.input.clone();
        let tx = self.tx.clone();

        std::thread::spawn(move || {
            let mask = merge_masks(&masks);
            let mut buf = Vec::new();
            mask.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                .unwrap();

            let response = zmq::request_response::<InpaintPayload>(
                "tcp://127.0.0.1:5559",
                zmq::Request::Inpaint {
                    prompt: input,
                    image_path: image_path,
                    mask: ByteBuf::from(buf),
                },
            )
            .unwrap();

            if let zmq::Response::Success(payload) = response {
                let _ = tx.send(image::load_from_memory(&payload.image).unwrap());
            }
        });
    }
}

impl super::Tool for InpaintTool {
    fn show(
        &mut self,
        ui: &mut Ui,
        canvas: &mut ImageCanvas,
        _mask_gallery: &mut MaskGallery,
        _undo_redo_manager: &mut UndoRedoManager,
    ) -> Option<LayerOperation> {
        if canvas.image_path.is_none() || canvas.selections.is_empty() {
            ui.disable();
        }

        if self.loading {
            ui.disable();
            ui.spinner();
        }

        ui.label("Inpaint Tool");

        ui.add(TextEdit::singleline(&mut self.input));

        let submit = ui.add(Button::new("Submit"));
        if submit.clicked() {
            self.fetch(canvas);
        }

        if let Ok(image) = self.rx.try_recv() {
            self.loading = false;
            // Instead of modifying canvas directly, return a layer operation
            // to create a new layer with the inpainted result
            let layer_name = format!("Inpainted - {}", self.input);
            return Some(LayerOperation::AddLayerWithImage(layer_name, image));
        }

        None
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
                // only write if source pixel is not black
                canvas.put_pixel(x, y, *pixel);
            }
        }
    }

    canvas
}
