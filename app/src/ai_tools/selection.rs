use std::sync::mpsc::{self, Receiver, Sender};

use eframe::egui::{Button, TextEdit, Ui};
use image::DynamicImage;
use serde_bytes::ByteBuf;

use crate::{ai_tools::selection, image_canvas::SharedCanvas};

use super::zmq;

pub struct SelectionTool {
    input: String,
    canvas: SharedCanvas,
}

impl SelectionTool {
    pub fn new(canvas: SharedCanvas) -> Self {
        Self {
            input: String::new(),
            canvas,
        }
    }

    fn fetch(&self, ctx: &eframe::egui::Context) {
        let image_path = {
            let canvas_ref = self.canvas.borrow();
            canvas_ref
                .image_path
                .as_ref()
                .unwrap()
                .to_string_lossy()
                .to_string()
        };

        let response = zmq::request_response(
            "tcp://127.0.0.1:5558",
            zmq::Request::ImageSelection {
                prompt: self.input.to_owned(),
                image_path: image_path.to_owned(),
            },
        )
        .unwrap();

        if let zmq::Response::Success(payload) = response {
            if let zmq::ImageSelectionPayload { masks } = payload {
                let selections: Vec<DynamicImage> = masks
                    .into_iter()
                    .filter_map(|mask_bytes| image::load_from_memory(&mask_bytes).ok())
                    .collect();

                self.canvas.borrow_mut().set_selections(selections, ctx);
            }
        }
    }
}

impl super::Tool for SelectionTool {
    fn show(&mut self, ui: &mut Ui) {
        ui.label("Selection Tool");

        ui.add(TextEdit::singleline(&mut self.input));

        let submit = ui.add(Button::new("Submit"));
        if submit.clicked() {
            self.fetch(ui.ctx());
        }

        let clear = ui.add(Button::new("Clear"));
        if clear.clicked() {
            self.canvas
                .borrow_mut()
                .set_selections(Vec::new(), ui.ctx());
        }
    }
}
