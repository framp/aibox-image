use eframe::egui::{Button, ComboBox, Ui};

use crate::{config::Config, image_canvas::ImageCanvas};

mod worker;

pub struct UpscaleTool {
    loading: bool,
    worker: worker::Worker,

    config: Config,
    selected_model_id: Option<usize>,
}

impl UpscaleTool {
    pub fn new(config: &Config) -> Self {
        let mut tool = Self {
            loading: false,
            worker: worker::Worker::new(),
            config: config.clone(),
            selected_model_id: None,
        };

        if let Some((id, first_model)) = tool.config.models.upscaling.iter().enumerate().next() {
            tool.loading = true;
            tool.selected_model_id = Some(id);
            // load the first model immediately
            tool.worker
                .load(first_model.kind.clone(), &tool.config.models.cache_dir);
        }

        tool
    }
}

impl super::Tool for UpscaleTool {
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.label("Upscale Tool");

        let mut clicked_model_id = None;
        ComboBox::from_label("Model3")
            .selected_text(
                self.selected_model_id
                    .and_then(|i| self.config.models.upscaling.get(i))
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
            self.loading = true;
            self.selected_model_id = Some(id);

            let model_kind = &self.config.models.upscaling[id].kind.clone();
            let cache_dir = self.config.models.cache_dir.clone();

            self.worker.load(model_kind.clone(), &cache_dir);
        }

        let can_submit =
            canvas.image_data.is_some() && self.selected_model_id.is_some() && !self.loading;

        let submit = ui.add_enabled(can_submit, Button::new("Submit"));
        let should_submit = submit.clicked();

        if should_submit && can_submit {
            self.loading = true;
            self.worker.upscale(canvas.image_data.as_ref().unwrap(), "");
        }

        if self.loading {
            ui.spinner();
        }

        if let Some(image) = self.worker.upscaled() {
            self.loading = false;
            canvas.set_image(image, ui.ctx());
        }

        if self.worker.loaded() {
            self.loading = false;
        }
    }
}
