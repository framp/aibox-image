use std::{
    io::Cursor,
    sync::mpsc::{self, Receiver, Sender},
};

use eframe::egui::{Button, TextEdit, Ui};
use image::{DynamicImage, GrayImage, Luma};
use serde_bytes::ByteBuf;

use crate::{ai_tools::zmq::InpaintPayload, image_canvas::ImageCanvas};

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

        let image_data = canvas.image_data.as_ref().unwrap();

        let mut image_buf = Vec::new();
        image_data
            .write_to(&mut Cursor::new(&mut image_buf), image::ImageFormat::Png)
            .unwrap();

        let masks = canvas
            .selections
            .iter()
            .filter(|s| s.visible)
            .map(|s| s.mask.clone())
            .collect();
        let input = self.input.clone();
        let tx = self.tx.clone();

        std::thread::spawn(move || {
            let mask = merge_masks(&masks);
            let mut mask_buf = Vec::new();
            mask.write_to(&mut Cursor::new(&mut mask_buf), image::ImageFormat::Png)
                .unwrap();

            let response = zmq::request_response::<InpaintPayload>(
                "tcp://127.0.0.1:5559",
                zmq::Request::Inpaint {
                    prompt: input,
                    image_bytes: ByteBuf::from(image_buf),
                    mask: ByteBuf::from(mask_buf),
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
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.label("Inpaint Tool");

        ui.add(TextEdit::singleline(&mut self.input));

        let has_visible_selections = canvas.selections.iter().any(|s| s.visible);
        let can_submit = canvas.image_data.is_some() && has_visible_selections && !self.loading;

        let submit = ui.add_enabled(can_submit, Button::new("Submit"));
        if submit.clicked() {
            self.fetch(canvas);
        }

        if self.loading {
            ui.spinner();
        }

        if let Ok(image) = self.rx.try_recv() {
            self.loading = false;
            canvas.set_image(image, ui.ctx());
            for selection in &mut canvas.selections {
                selection.visible = false;
            }
        }
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
