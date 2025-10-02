use eframe::egui::{Button, ComboBox, TextEdit, Ui};
use tokio::sync::mpsc::Sender;

use crate::{
    config::Config,
    error::Error,
    image_canvas::ImageCanvas,
    worker::{ErrorChan, WorkerTrait},
};

mod worker;

pub struct InpaintTool {
    input: String,

    worker: worker::Worker,

    config: Config,
    selected_model: Option<String>,
}

impl InpaintTool {
    pub fn new(config: &Config, tx_err: ErrorChan) -> Self {
        let tool = Self {
            input: String::new(),
            worker: worker::Worker::new(tx_err),
            config: config.clone(),
            selected_model: None,
        };

        if let Some(first_model) = tool.config.models.inpainting.iter().next() {
            // load the first model immediately
            tool.worker
                .load_model(first_model, &tool.config.models.cache_dir);
        }

        tool
    }

    fn ui_model_selection(&mut self, ui: &mut Ui, enabled: bool) {
        let mut clicked_model_id = None;

        ui.add_enabled_ui(enabled, |ui| {
            ComboBox::from_label("Model")
                .selected_text(self.selected_model.as_deref().unwrap_or("Select..."))
                .show_ui(ui, |ui| {
                    for (id, obj) in self.config.models.inpainting.iter().enumerate() {
                        if ui.selectable_label(false, &obj.name).clicked() {
                            clicked_model_id = Some(id);
                        }
                    }
                });
        });

        if let Some(id) = clicked_model_id {
            self.worker.load_model(
                &self.config.models.inpainting[id],
                &self.config.models.cache_dir,
            );
        }
    }

    fn ui_inpainting(&mut self, ui: &mut Ui, canvas: &ImageCanvas, enabled: bool) {
        let text_edit_response = ui.add(TextEdit::singleline(&mut self.input));

        let submit = ui.add_enabled(enabled, Button::new("Submit"));
        let should_submit = submit.clicked()
            || (text_edit_response.lost_focus()
                && ui.input(|i| i.key_pressed(eframe::egui::Key::Enter)));

        if should_submit && enabled {
            self.worker.inpaint(
                canvas.image.as_ref().unwrap().image(),
                canvas
                    .selections
                    .iter()
                    .filter(|s| s.visible)
                    .map(|s| s.mask.clone())
                    .collect(),
                &self.input,
            );
        }
    }
}

impl super::Tool for InpaintTool {
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.push_id("inpaint", |ui| {
            ui.label("Inpaint Tool");

            let has_visible_selections = canvas.selections.iter().any(|s| s.visible);
            let ui_enabled = canvas.image.is_some()
                && has_visible_selections
                && self.selected_model.is_some()
                && !self.worker.is_processing();

            self.ui_model_selection(ui, ui_enabled);
            self.ui_inpainting(ui, canvas, ui_enabled);

            if self.worker.is_processing() {
                ui.spinner();
            }

            if let Some(image) = self.worker.inpainted() {
                canvas.set_image(image, ui.ctx());
                for selection in &mut canvas.selections {
                    selection.visible = false;
                }
            }

            if let Some(model) = self.worker.model_loaded() {
                self.selected_model = Some(model);
            }
        });
    }
}
