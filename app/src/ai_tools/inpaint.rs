use std::{
    io::Cursor,
    sync::mpsc::{self, Receiver, Sender},
};

use eframe::egui::{Button, TextEdit, Ui};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use serde_bytes::ByteBuf;

use crate::{ai_tools::zmq::InpaintPayload, image_canvas::SharedCanvas};

use super::zmq;

pub struct InpaintTool {
    input: String,
    canvas: SharedCanvas,

    loading: bool,
    tx: Sender<DynamicImage>,
    rx: Receiver<DynamicImage>,
}

impl InpaintTool {
    pub fn new(canvas: SharedCanvas) -> Self {
        let (tx, rx) = mpsc::channel();

        Self {
            input: String::new(),
            canvas,
            loading: false,
            tx,
            rx,
        }
    }

    fn fetch(&mut self) {
        self.loading = true;

        let (image_path, masks) = {
            let canvas_ref = self.canvas.borrow();
            let image_path = canvas_ref
                .image_path
                .as_ref()
                .unwrap()
                .to_string_lossy()
                .to_string();

            (image_path, canvas_ref.selections.clone())
        };
        let input = self.input.clone();
        let tx = self.tx.clone();

        std::thread::spawn(move || {
            let mask = merge_selections(&masks);
            let mask = to_bw_mask(&mask);
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
    fn show(&mut self, ui: &mut Ui) {
        if self.canvas.borrow().image_path.is_none() || self.canvas.borrow().selections.is_empty() {
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
            self.fetch();
        }

        if let Ok(image) = self.rx.try_recv() {
            self.loading = false;
            self.canvas.borrow_mut().set_image(image, None, ui.ctx());
        }
    }
}

fn merge_selections(images: &Vec<DynamicImage>) -> DynamicImage {
    if images.is_empty() {
        return DynamicImage::new_rgba8(1, 1);
    }

    let width = images.iter().map(|img| img.width()).max().unwrap();
    let height = images.iter().map(|img| img.height()).max().unwrap();

    let mut canvas: RgbaImage = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));

    for img in images {
        let rgba_img = img.to_rgba8();
        for (x, y, pixel) in rgba_img.enumerate_pixels() {
            if pixel[3] != 0 {
                // only write if source pixel is non-transparent
                canvas.put_pixel(x, y, *pixel);
            }
        }
    }

    DynamicImage::ImageRgba8(canvas)
}

/// Convert a DynamicImage into a black-and-white mask:
/// - Non-transparent pixels → white (255,255,255,255)  
/// - Transparent pixels → black (0,0,0,255)
fn to_bw_mask(image: &DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    let rgba = image.to_rgba8(); // ensure RGBA8
    let mut new_img: RgbaImage = RgbaImage::new(width, height);

    for (x, y, pixel) in rgba.enumerate_pixels() {
        let alpha = pixel[3];
        let new_pixel = if alpha != 0 {
            Rgba([255, 255, 255, 255])
        } else {
            Rgba([0, 0, 0, 255])
        };
        new_img.put_pixel(x, y, new_pixel);
    }

    DynamicImage::ImageRgba8(new_img)
}
