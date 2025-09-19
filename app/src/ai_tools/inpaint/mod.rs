use eframe::egui::{Button, ComboBox, TextEdit, Ui};

use crate::{config::Config, image_canvas::ImageCanvas};

mod worker;

pub struct InpaintTool {
    input: String,

    loading: bool,
    worker: worker::Worker,

    config: Config,
    selected_model_id: Option<usize>,
}

impl InpaintTool {
    pub fn new(config: &Config) -> Self {
        let mut tool = Self {
            input: String::new(),
            loading: false,
            worker: worker::Worker::new(),
            config: config.clone(),
            selected_model_id: None,
        };

        if let Some((id, first_model)) = tool.config.models.inpainting.iter().enumerate().next() {
            tool.loading = true;
            tool.selected_model_id = Some(id);
            // load the first model immediately
            tool.worker
                .load(first_model.kind.clone(), &tool.config.models.cache_dir);
        }

        tool
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
            self.loading = true;
            self.selected_model_id = Some(id);

            let model_kind = &self.config.models.inpainting[id].kind.clone();
            let cache_dir = self.config.models.cache_dir.clone();

            self.worker.load(model_kind.clone(), &cache_dir);
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
            self.loading = true;
            self.worker.inpaint(
                canvas.image_data.as_ref().unwrap(),
                canvas
                    .selections
                    .iter()
                    .filter(|s| s.visible)
                    .map(|s| s.mask.clone())
                    .collect(),
                &self.input,
            );
        }

        if self.loading {
            ui.spinner();
        }

        if let Some(image) = self.worker.inpainted() {
            self.loading = false;
            canvas.set_image(image, ui.ctx());
            for selection in &mut canvas.selections {
                selection.visible = false;
            }
        }

        if self.worker.loaded() {
            self.loading = false;
        }
    }
}
