use std::{
    io::Cursor,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
};

use eframe::egui::{Button, ComboBox, TextEdit, Ui};
use image::{DynamicImage, GrayImage, Luma};
use serde_bytes::ByteBuf;

use crate::{
    ai_tools::zmq::InpaintPayload,
    config::{Config, InpaintingModel},
    image_canvas::ImageCanvas,
};

use super::zmq;

pub struct InpaintTool {
    input: String,

    loading: bool,
    tx: Sender<DynamicImage>,
    rx: Receiver<DynamicImage>,

    load_tx: Sender<()>,
    load_rx: Receiver<()>,

    config: Config,
    selected_model_id: Option<usize>,
}

impl InpaintTool {
    pub fn new(config: &Config) -> Self {
        let (tx, rx) = mpsc::channel();
        let (load_tx, load_rx) = mpsc::channel();

        let mut tool = Self {
            input: String::new(),
            loading: false,
            tx,
            rx,
            load_tx,
            load_rx,
            config: config.clone(),
            selected_model_id: None,
        };

        if let Some((id, first_model)) = tool.config.models.inpainting.iter().enumerate().next() {
            tool.selected_model_id = Some(id);
            // load the first model immediately
            let cache_dir = tool.config.models.cache_dir.clone();
            tool.load(&first_model.kind.clone(), &cache_dir);
        }

        tool
    }

    fn load(&mut self, model: &InpaintingModel, cache_dir: &PathBuf) {
        self.loading = true;

        let model = model.clone();
        let cache_dir = cache_dir.to_str().unwrap().to_owned();
        let tx = self.load_tx.clone();

        std::thread::spawn(move || {
            let response = zmq::request_response::<()>(
                "tcp://127.0.0.1:5559",
                zmq::Request::Load {
                    cache_dir,
                    model: zmq::ModelKind::Inpainting(model),
                },
            )
            .unwrap();

            tx.send(())
        });
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
            .map(|s| s.applied_mask.clone())
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

        let mut clicked_model_id = None;
        ComboBox::from_label("Model2")
            .selected_text(
                self.selected_model_id
                    .and_then(|i| self.config.models.inpainting.get(i))
                    .map(|obj| obj.name.as_str())
                    .unwrap_or("Select..."),
            )
            .show_ui(ui, |ui| {
                for (id, obj) in self.config.models.inpainting.iter().enumerate() {
                    if ui
                        .selectable_label(self.selected_model_id == Some(id), &obj.name)
                        .clicked()
                    {
                        clicked_model_id = Some(id);
                    }
                }
            });

        if let Some(id) = clicked_model_id {
            self.selected_model_id = Some(id);

            let model_kind = &self.config.models.inpainting[id].kind.clone();
            let cache_dir = self.config.models.cache_dir.clone();

            self.load(model_kind, &cache_dir);
        }

        let text_edit_response = ui.add(TextEdit::singleline(&mut self.input));

        let has_visible_selections = canvas.selections.iter().any(|s| s.visible);
        let can_submit = canvas.image_data.is_some()
            && has_visible_selections
            && self.selected_model_id.is_some()
            && !self.loading;

        let submit = ui.add_enabled(can_submit, Button::new("Submit"));
        let should_submit = submit.clicked()
            || (text_edit_response.lost_focus()
                && ui.input(|i| i.key_pressed(eframe::egui::Key::Enter)));

        if should_submit && can_submit {
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

        if let Ok(_) = self.load_rx.try_recv() {
            self.loading = false;
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
