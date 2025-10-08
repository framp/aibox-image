use eframe::egui::{Button, ComboBox, Ui};

use crate::{config::Config, image_canvas::ImageCanvas, worker::ErrorChan};

mod worker;

pub struct UpscaleTool {
    loading: bool,
    worker: worker::Worker,

    config: Config,
    selected_model: Option<String>,
}

impl UpscaleTool {
    pub fn new(config: &Config, tx_err: ErrorChan) -> Self {
        let mut tool = Self {
            loading: false,
            worker: worker::Worker::new(tx_err),
            config: config.clone(),
            selected_model: None,
        };

        if let Some(first_model) = tool.config.models.upscaling.first() {
            tool.loading = true;
            // load the first model immediately
            tool.worker
                .load(first_model.kind.clone(), &tool.config.models.cache_dir);
        }

        tool
    }
}

impl super::Tool for UpscaleTool {
    fn name(&self) -> &str {
        "Upscale"
    }

    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.push_id("upscale", |ui| {
            ui.label("Upscale Tool");

            let mut clicked_model_id = None;
            ComboBox::from_label("Model")
                .selected_text(self.selected_model.as_deref().unwrap_or("Select..."))
                .show_ui(ui, |ui| {
                    for (id, obj) in self.config.models.upscaling.iter().enumerate() {
                        if ui.selectable_label(false, &obj.name).clicked() {
                            clicked_model_id = Some(id);
                        }
                    }
                });

            if let Some(id) = clicked_model_id {
                self.loading = true;

                let model_kind = &self.config.models.upscaling[id].kind.clone();
                let cache_dir = self.config.models.cache_dir.clone();

                self.worker.load(model_kind.clone(), &cache_dir);
            }

            let can_submit =
                canvas.image.is_some() && self.selected_model.is_some() && !self.loading;

            let submit = ui.add_enabled(can_submit, Button::new("Submit"));
            let should_submit = submit.clicked();

            if should_submit && can_submit {
                self.loading = true;
                self.worker
                    .upscale(canvas.image.as_ref().unwrap().image(), "");
            }

            if self.loading {
                ui.spinner();
            }

            if let Some(image) = self.worker.upscaled() {
                self.loading = false;
                canvas.set_image_with_history(image, ui.ctx(), crate::history::Action::Upscale);
            }

            if self.worker.loaded() {
                self.loading = false;
            }
        });
    }
}
