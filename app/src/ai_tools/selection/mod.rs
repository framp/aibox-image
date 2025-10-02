use eframe::egui::{
    Button, Checkbox, CollapsingHeader, ComboBox, DragValue, Image, Slider, TextEdit, Ui,
};
use image::{GrayImage, Luma};

use crate::{
    config::Config,
    image_canvas::{ImageCanvas, Selection},
    worker::{ErrorChan, WorkerTrait},
};

mod worker;

pub struct SelectionTool {
    input: String,
    threshold: f32,
    worker: worker::Worker,
    config: Config,
    selected_model: Option<String>,
}

impl SelectionTool {
    pub fn new(config: &Config, tx_err: ErrorChan) -> Self {
        let tool = Self {
            input: String::new(),
            threshold: 0.5,
            worker: worker::Worker::new(tx_err),
            config: config.clone(),
            selected_model: None,
        };

        if let Some(first_model) = tool.config.models.selection.first() {
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
                    for (id, obj) in self.config.models.selection.iter().enumerate() {
                        if ui.selectable_label(false, &obj.name).clicked() {
                            clicked_model_id = Some(id);
                        }
                    }
                });
        });

        if let Some(id) = clicked_model_id {
            self.worker.load_model(
                &self.config.models.selection[id],
                &self.config.models.cache_dir,
            );
        }
    }

    fn ui_image_selection(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas, enabled: bool) {
        let text_edit_response = ui.add_enabled(enabled, TextEdit::singleline(&mut self.input));
        ui.horizontal(|ui| {
            ui.label("Threshold:");
            ui.add_enabled(
                enabled,
                Slider::new(&mut self.threshold, 0.0..=1.0).step_by(0.01),
            );
        });

        ui.horizontal(|ui| {
            let submit = ui.add_enabled(enabled, Button::new("Submit"));
            let should_submit = submit.clicked()
                || (text_edit_response.lost_focus()
                    && ui.input(|i| i.key_pressed(eframe::egui::Key::Enter)));

            if should_submit && enabled {
                self.worker.image_selection(
                    canvas.image.as_ref().unwrap().image(),
                    &self.input,
                    self.threshold,
                );
            }

            let select_canvas = ui.add_enabled(enabled, Button::new("Select canvas"));
            if select_canvas.clicked() {
                let size = canvas.image.as_ref().unwrap().size();
                canvas.selections.push(Selection::from_mask(
                    ui.ctx(),
                    GrayImage::from_pixel(size.x as u32, size.y as u32, Luma([255])),
                ));
            }
        });
    }

    fn ui_selections(&self, ui: &mut Ui, canvas: &mut ImageCanvas, enabled: bool) {
        CollapsingHeader::new("Selections")
            .default_open(true)
            .show(ui, |ui| {
                let mut to_remove: Option<usize> = None;
                let mut to_invert: Option<usize> = None;
                let mut visibility_updates: Vec<(usize, bool)> = Vec::new();
                let mut parameter_updates: Vec<(usize, i32, u32)> = Vec::new();

                let remove_all = ui.add_enabled(enabled, Button::new("Remove All"));
                if remove_all.clicked() {
                    canvas.selections = Vec::new();
                }

                for (i, sel) in canvas.selections.iter().enumerate() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            let size = sel.overlay_texture.size_vec2();
                            let max_side = 128.0;
                            let scale = (max_side / size.x.max(size.y)).min(1.0);
                            let thumb_size = size * scale;

                            let mut visible = sel.visible;
                            let response =
                                ui.add_enabled(enabled, Checkbox::without_text(&mut visible));
                            if response.changed() {
                                visibility_updates.push((i, visible));
                            }

                            ui.add_enabled(
                                enabled,
                                Image::new(&sel.mask_texture).fit_to_exact_size(thumb_size),
                            );

                            ui.vertical(|ui| {
                                if ui.add_enabled(enabled, Button::new("🔄")).clicked() {
                                    to_invert = Some(i);
                                }

                                if ui.add_enabled(enabled, Button::new("🗑")).clicked() {
                                    to_remove = Some(i);
                                }
                            });
                        });

                        ui.horizontal(|ui| {
                            ui.label("Growth:");
                            let mut growth = sel.growth;
                            let growth_response =
                                ui.add_enabled(enabled, DragValue::new(&mut growth).speed(1.0));

                            ui.label("Blur:");
                            let mut blur = sel.blur;
                            let blur_response = ui.add_enabled(
                                enabled,
                                DragValue::new(&mut blur).speed(1.0).range(0..=100),
                            );

                            if growth_response.changed() || blur_response.changed() {
                                parameter_updates.push((i, growth, blur));
                            }
                        });
                    });
                }

                for (i, visible) in visibility_updates {
                    canvas.selections[i].visible = visible;
                }

                for (i, growth, blur) in parameter_updates {
                    canvas.selections[i].growth = growth;
                    canvas.selections[i].blur = blur;
                    canvas.selections[i].update_applied_mask(ui.ctx());
                }

                if let Some(i) = to_invert {
                    let old = &canvas.selections[i];
                    canvas.selections[i] =
                        Selection::from_mask(ui.ctx(), invert_mask(&old.original_mask));
                }

                if let Some(i) = to_remove {
                    canvas.selections.remove(i);
                }
            });
    }
}

impl super::Tool for SelectionTool {
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.push_id("selection", |ui| {
            ui.label("Selection Tool");

            let ui_enabled = canvas.image.is_some()
                && self.selected_model.is_some()
                && !self.worker.is_processing();

            self.ui_model_selection(ui, ui_enabled);

            if self.worker.is_processing() {
                ui.spinner();
            }

            self.ui_image_selection(ui, canvas, ui_enabled);
            self.ui_selections(ui, canvas, ui_enabled);

            if let Some(model) = self.worker.model_loaded() {
                self.selected_model = Some(model);
            }

            if let Some(selections) = self.worker.selections() {
                canvas.selections = selections
                    .into_iter()
                    .enumerate()
                    .map(|(i, img)| {
                        let mut selection = Selection::from_mask(ui.ctx(), img);
                        // Only enable the first mask, disable all others
                        selection.visible = i == 0;
                        selection
                    })
                    .collect();
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
