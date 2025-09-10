use std::sync::mpsc::{self, Receiver, Sender};

use eframe::egui::{Button, CollapsingHeader, Image, Slider, TextEdit, Ui};
use image::DynamicImage;

use crate::image_canvas::SharedCanvas;

use super::zmq;

pub struct SelectionTool {
    input: String,
    threshold: f32,
    canvas: SharedCanvas,

    loading: bool,
    tx: Sender<Vec<DynamicImage>>,
    rx: Receiver<Vec<DynamicImage>>,
}

impl SelectionTool {
    pub fn new(canvas: SharedCanvas) -> Self {
        let (tx, rx) = mpsc::channel();

        Self {
            input: String::new(),
            threshold: 0.5,
            canvas,
            loading: false,
            tx,
            rx,
        }
    }

    fn fetch(&mut self) {
        self.loading = true;

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
        let threshold = self.threshold;
        let tx = self.tx.clone();

        std::thread::spawn(move || {
            let response = zmq::request_response(
                "tcp://127.0.0.1:5558",
                zmq::Request::ImageSelection {
                    prompt: input,
                    image_path: image_path,
                    threshold,
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
        if self.canvas.borrow().image_path.is_none() {
            ui.disable();
        }

        if self.loading {
            ui.disable();
            ui.spinner();
        }

        ui.label("Selection Tool");

        ui.add(TextEdit::singleline(&mut self.input));

        ui.horizontal(|ui| {
            ui.label("Threshold:");
            ui.add(Slider::new(&mut self.threshold, 0.0..=1.0).step_by(0.01));
        });

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
            self.loading = false;
            self.canvas
                .borrow_mut()
                .set_selections(selections, ui.ctx());
        }

        CollapsingHeader::new("Selections")
            .default_open(true)
            .show(ui, |ui| {
                let mut to_remove: Option<usize> = None;

                for (i, tex) in self.canvas.borrow().selections.iter().enumerate() {
                    ui.horizontal(|ui| {
                        // Show the texture as a small thumbnail
                        let size = tex.size_vec2();
                        let max_side = 128.0;
                        let scale = (max_side / size.x.max(size.y)).min(1.0);
                        let thumb_size = size * scale;

                        ui.add(Image::new(tex).fit_to_exact_size(thumb_size));

                        if ui.button("Remove").clicked() {
                            to_remove = Some(i);
                        }
                    });
                }

                if let Some(i) = to_remove {
                    self.canvas.borrow_mut().selections.remove(i);
                }
            });
    }
}
