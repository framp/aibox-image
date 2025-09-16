use std::sync::mpsc::{self, Receiver, Sender};

use eframe::egui::{Button, Checkbox, CollapsingHeader, Image, Slider, TextEdit, Ui};
use image::{DynamicImage, GrayImage, Luma};

use crate::image_canvas::{ImageCanvas, Selection};

use super::zmq;

pub struct SelectionTool {
    input: String,
    threshold: f32,
    loading: bool,
    tx: Sender<Vec<GrayImage>>,
    rx: Receiver<Vec<GrayImage>>,
}

impl SelectionTool {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();

        Self {
            input: String::new(),
            threshold: 0.5,
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
            .write_to(
                &mut std::io::Cursor::new(&mut image_buf),
                image::ImageFormat::Png,
            )
            .unwrap();

        let input = self.input.clone();
        let threshold = self.threshold;
        let tx = self.tx.clone();

        std::thread::spawn(move || {
            let response = zmq::request_response(
                "tcp://127.0.0.1:5558",
                zmq::Request::ImageSelection {
                    prompt: input,
                    image_bytes: serde_bytes::ByteBuf::from(image_buf),
                    threshold,
                },
            )
            .unwrap();

            if let zmq::Response::Success(payload) = response {
                if let zmq::ImageSelectionPayload { masks } = payload {
                    let selections = masks
                        .into_iter()
                        .filter_map(|mask_bytes| {
                            image::load_from_memory(&mask_bytes)
                                .ok()
                                .map(|img: DynamicImage| img.into_luma8())
                        })
                        .collect::<Vec<_>>();

                    let _ = tx.send(selections);
                }
            }
        });
    }
}

impl super::Tool for SelectionTool {
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.label("Selection Tool");

        ui.add(TextEdit::singleline(&mut self.input));

        ui.horizontal(|ui| {
            ui.label("Threshold:");
            ui.add(Slider::new(&mut self.threshold, 0.0..=1.0).step_by(0.01));
        });

        let can_submit = canvas.image_data.is_some() && !self.loading;

        let submit = ui.add_enabled(can_submit, Button::new("Submit"));
        if submit.clicked() {
            self.fetch(canvas);
        }

        let clear = ui.add(Button::new("Clear"));
        if clear.clicked() {
            canvas.selections = Vec::new();
        }

        let select_all = ui.add(Button::new("Select canvas"));
        if select_all.clicked() {
            let size = canvas.image_size;
            canvas.selections.push(Selection::from_mask(
                ui.ctx(),
                GrayImage::from_pixel(size.x as u32, size.y as u32, Luma([255])),
            ));
        }

        if self.loading {
            ui.spinner();
        }

        if let Ok(selections) = self.rx.try_recv() {
            self.loading = false;
            canvas.selections = selections
                .into_iter()
                .map(|img| Selection::from_mask(ui.ctx(), img))
                .collect();
        }

        CollapsingHeader::new("Selections")
            .default_open(true)
            .show(ui, |ui| {
                let mut to_remove: Option<usize> = None;
                let mut to_invert: Option<usize> = None;
                let mut visibility_updates: Vec<(usize, bool)> = Vec::new();

                for (i, sel) in canvas.selections.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let size = sel.overlay_texture.size_vec2();
                        let max_side = 128.0;
                        let scale = (max_side / size.x.max(size.y)).min(1.0);
                        let thumb_size = size * scale;

                        let mut visible = sel.visible;
                        let response = ui.add(Checkbox::without_text(&mut visible));
                        if response.changed() {
                            visibility_updates.push((i, visible));
                        }

                        ui.add(Image::new(&sel.mask_texture).fit_to_exact_size(thumb_size));

                        if ui.button("Invert").clicked() {
                            to_invert = Some(i);
                        }

                        if ui.button("Remove").clicked() {
                            to_remove = Some(i);
                        }
                    });
                }

                for (i, visible) in visibility_updates {
                    canvas.selections[i].visible = visible;
                }

                if let Some(i) = to_invert {
                    let old = &canvas.selections[i];
                    canvas.selections[i] = Selection::from_mask(ui.ctx(), invert_mask(&old.mask));
                }

                if let Some(i) = to_remove {
                    canvas.selections.remove(i);
                }
            });
    }
}

fn invert_mask(image: &GrayImage) -> GrayImage {
    let black_pixel = Luma([0]);
    let white_pixel = Luma([255]);

    let (width, height) = image.dimensions();
    let mut new_img = GrayImage::new(width, height);

    for (x, y, p) in image.enumerate_pixels() {
        let alpha = p[0];
        let new_pixel = if alpha == 0 { white_pixel } else { black_pixel };
        new_img.put_pixel(x, y, new_pixel);
    }

    new_img
}
