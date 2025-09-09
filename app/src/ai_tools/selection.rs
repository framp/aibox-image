use std::sync::mpsc::{self, Receiver, Sender};

use eframe::egui::{Button, TextEdit, Ui};
use image::DynamicImage;

use crate::image_canvas::SharedCanvas;

use super::zmq;

pub struct SelectionTool {
    input: String,
    canvas: SharedCanvas,
    tx: Sender<Vec<DynamicImage>>,
    rx: Receiver<Vec<DynamicImage>>,
}

impl SelectionTool {
    pub fn new(canvas: SharedCanvas) -> Self {
        let (tx, rx) = mpsc::channel();

        Self {
            input: String::new(),
            canvas,
            tx,
            rx,
        }
    }

    fn fetch(&self) {
        let image_path = {
            let canvas_ref = self.canvas.borrow();
            canvas_ref
                .image_path
                .as_ref()
                .unwrap()
                .to_string_lossy()
                .to_string()
        };
        let input = self.input.clone();
        let tx = self.tx.clone();

        std::thread::spawn(move || {
            let response = zmq::request_response(
                "tcp://127.0.0.1:5558",
                zmq::Request::ImageSelection {
                    prompt: input,
                    image_path: image_path,
                },
            )
            .unwrap();

            if let zmq::Response::Success(payload) = response {
                if let zmq::ImageSelectionPayload { masks } = payload {
                    let selections: Vec<DynamicImage> = masks
                        .into_iter()
                        .filter_map(|mask_bytes| image::load_from_memory(&mask_bytes).ok())
                        .collect();

                    let _ = tx.send(selections);
                }
            }
        });
    }
}

impl super::Tool for SelectionTool {
    fn show(&mut self, ui: &mut Ui) {
        ui.label("Selection Tool");

        ui.add(TextEdit::singleline(&mut self.input));

        let submit = ui.add(Button::new("Submit"));
        if submit.clicked() {
            self.fetch();
        }

        let clear = ui.add(Button::new("Clear"));
        if clear.clicked() {
            self.canvas
                .borrow_mut()
                .set_selections(Vec::new(), ui.ctx());
        }

        if let Ok(selections) = self.rx.try_recv() {
            self.canvas
                .borrow_mut()
                .set_selections(selections, ui.ctx());
        }
    }
}
