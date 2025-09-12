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

        let image = canvas.image.clone();
        let masks = canvas.selections.iter().map(|s| s.mask.clone()).collect();
        let prompt = self.input.clone();
        let tx = self.tx.clone();

        std::thread::spawn(move || {
            let mut image_buf = Vec::new();
            image
                .write_to(&mut Cursor::new(&mut image_buf), image::ImageFormat::Png)
                .unwrap();

            let mask = merge_masks(&masks);
            let mut mask_buf = Vec::new();
            mask.write_to(&mut Cursor::new(&mut mask_buf), image::ImageFormat::Png)
                .unwrap();

            let response = zmq::request_response::<InpaintPayload>(
                "tcp://127.0.0.1:5559",
                zmq::Request::Inpaint {
                    prompt,
                    image: ByteBuf::from(image_buf),
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
        if canvas.selections.is_empty() {
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
            canvas.update_image(image, ui.ctx());
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
                // only write if source pixel is not black
                canvas.put_pixel(x, y, *pixel);
            }
        }
    }

    canvas
}
