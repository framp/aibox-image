use std::sync::mpsc::{self, Receiver, Sender};

use eframe::egui::{Button, Checkbox, CollapsingHeader, Image, TextEdit, Ui};
use image::DynamicImage;

use crate::image_canvas::ImageCanvas;

use super::zmq;

pub struct SelectionTool {
    input: String,
    loading: bool,
    tx: Sender<Vec<DynamicImage>>,
    rx: Receiver<Vec<DynamicImage>>,
}

impl SelectionTool {
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
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        if canvas.image_path.is_none() {
            ui.disable();
        }

        if self.loading {
            ui.disable();
            ui.spinner();
        }

        ui.label("Selection Tool");

        ui.add(TextEdit::singleline(&mut self.input));

        let submit = ui.add(Button::new("Submit"));
        if submit.clicked() {
            self.fetch(canvas);
        }

        let clear = ui.add(Button::new("Clear"));
        if clear.clicked() {
            canvas.set_selections(Vec::new(), ui.ctx());
        }

        if let Ok(selections) = self.rx.try_recv() {
            self.loading = false;
            canvas.set_selections(selections, ui.ctx());
        }

        CollapsingHeader::new("Selections")
            .default_open(true)
            .show(ui, |ui| {
                let mut to_remove: Option<usize> = None;
                let mut visibility_updates: Vec<(usize, bool)> = Vec::new();

                for (i, sel) in canvas.selections.iter().enumerate() {
                    ui.horizontal(|ui| {
                        // Show the texture as a small thumbnail
                        let size = sel.texture.size_vec2();
                        let max_side = 128.0;
                        let scale = (max_side / size.x.max(size.y)).min(1.0);
                        let thumb_size = size * scale;

                        let mut visible = sel.visible;
                        let response = ui.add(Checkbox::without_text(&mut visible));
                        if response.changed() {
                            visibility_updates.push((i, visible));
                        }

                        ui.add(Image::new(&sel.texture).fit_to_exact_size(thumb_size));

                        if ui.button("Remove").clicked() {
                            to_remove = Some(i);
                        }
                    });
                }

                for (i, visible) in visibility_updates {
                    canvas.selections[i].visible = visible;
                }

                if let Some(i) = to_remove {
                    canvas.selections.remove(i);
                }
            });
    }
}
