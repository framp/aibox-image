use eframe::egui::{
    Button, Checkbox, CollapsingHeader, ComboBox, DragValue, Image, Slider, TextEdit, Ui,
};
use image::{GrayImage, Luma};

use crate::{
    config::Config,
    image_canvas::{ImageCanvas, Selection},
    worker::WorkerTrait,
};

mod worker;

pub struct SelectionTool {
    input: String,
    threshold: f32,
    worker: worker::Worker,
    config: Config,
    selected_model: Option<String>,
    current_tool: ToolMode,
    brush_size: f32,
    show_brush_settings: bool,
    selected_layer: Option<usize>,
    undo_stack: Vec<UndoAction>,
    redo_stack: Vec<UndoAction>,
}


#[derive(Clone, Copy, PartialEq)]
enum ToolMode {
    Pan,
    AddBrush,
    DeleteBrush,
}

#[derive(Clone)]
enum UndoAction {
    BrushStroke {
        layer_index: usize,
        previous_mask: GrayImage,
    },
    ToolSubmit {
        previous_selections: Vec<Selection>,
    },
}

impl SelectionTool {
    pub fn new(config: &Config) -> Self {
        let tool = Self {
            input: String::new(),
            threshold: 0.5,
            worker: worker::Worker::new(),
            config: config.clone(),
            selected_model: None,
            current_tool: ToolMode::Pan,
            brush_size: 20.0,
            show_brush_settings: false,
            selected_layer: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        };

        if let Some(first_model) = tool.config.models.selection.iter().next() {
            // load the first model immediately
            tool.worker
                .load_model(&first_model, &tool.config.models.cache_dir);
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
                // Switch to add brush mode after submitting selection
                self.current_tool = ToolMode::AddBrush;
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

    fn ui_selections(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas, enabled: bool) {
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


    fn ui_layer_selection(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas, enabled: bool) {
        ui.horizontal(|ui| {
            ui.label("Edit Layer:");
            ui.add_enabled_ui(enabled, |ui| {
                ComboBox::from_id_salt("selection_layer_selector")
                    .selected_text(
                        self.selected_layer
                            .map(|i| format!("Layer {}", i + 1))
                            .unwrap_or_else(|| "Select...".to_string()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, _) in canvas.selections.iter().enumerate() {
                            if ui
                                .selectable_label(
                                    self.selected_layer == Some(i),
                                    format!("Layer {}", i + 1),
                                )
                                .clicked()
                            {
                                self.selected_layer = Some(i);
                            }
                        }
                    });
            });

        });

        // Auto-select first layer if none selected and layers exist
        if self.selected_layer.is_none() && !canvas.selections.is_empty() {
            self.selected_layer = Some(0);
        }

        // Clear selection if layer no longer exists
        if let Some(idx) = self.selected_layer {
            if idx >= canvas.selections.len() {
                self.selected_layer = None;
            }
        }
    }


    fn undo(&mut self, canvas: &mut ImageCanvas, ctx: &eframe::egui::Context) {
        if let Some(action) = self.undo_stack.pop() {
            match action {
                UndoAction::BrushStroke { layer_index, previous_mask } => {
                    if layer_index < canvas.selections.len() {
                        let current_mask = canvas.selections[layer_index].original_mask.clone();
                        self.redo_stack.push(UndoAction::BrushStroke {
                            layer_index,
                            previous_mask: current_mask,
                        });

                        canvas.selections[layer_index] = Selection::from_mask(ctx, previous_mask);
                    }
                }
                UndoAction::ToolSubmit { previous_selections } => {
                    let current_selections = canvas.selections.clone();
                    self.redo_stack.push(UndoAction::ToolSubmit {
                        previous_selections: current_selections,
                    });
                    canvas.selections = previous_selections;
                }
            }
        }
    }

    fn redo(&mut self, canvas: &mut ImageCanvas, ctx: &eframe::egui::Context) {
        if let Some(action) = self.redo_stack.pop() {
            match action {
                UndoAction::BrushStroke { layer_index, previous_mask } => {
                    if layer_index < canvas.selections.len() {
                        let current_mask = canvas.selections[layer_index].original_mask.clone();
                        self.undo_stack.push(UndoAction::BrushStroke {
                            layer_index,
                            previous_mask: current_mask,
                        });

                        canvas.selections[layer_index] = Selection::from_mask(ctx, previous_mask);
                    }
                }
                UndoAction::ToolSubmit { previous_selections } => {
                    let current_selections = canvas.selections.clone();
                    self.undo_stack.push(UndoAction::ToolSubmit {
                        previous_selections: current_selections,
                    });
                    canvas.selections = previous_selections;
                }
            }
        }
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

            // Handle brush interactions when using brush tools
            if matches!(self.current_tool, ToolMode::AddBrush | ToolMode::DeleteBrush) && self.selected_layer.is_some() {
                canvas.brush_enabled = true;

                let brush_size = self.brush_size;
                let current_tool = self.current_tool;
                let selected_layer = self.selected_layer.unwrap();

                canvas.brush_paint_callback = Some(Box::new(move |selections, pos, last_pos, image_rect, ctx| {
                    if selected_layer < selections.len() {
                        let image_size = if let Some(image) = &selections.get(0) {
                            [image.original_mask.width(), image.original_mask.height()]
                        } else {
                            return;
                        };

                        let rect_size = image_rect.size();
                        let rel_x = (pos.x - image_rect.min.x) / rect_size.x;
                        let rel_y = (pos.y - image_rect.min.y) / rect_size.y;

                        if rel_x < 0.0 || rel_x > 1.0 || rel_y < 0.0 || rel_y > 1.0 {
                            return;
                        }

                        let img_x = (rel_x * image_size[0] as f32) as i32;
                        let img_y = (rel_y * image_size[1] as f32) as i32;

                        let radius = brush_size / 2.0;
                        let color = match current_tool {
                            ToolMode::AddBrush => Luma([255u8]),
                            ToolMode::DeleteBrush => Luma([0u8]),
                            _ => return, // Should not happen
                        };

                        let mut mask = selections[selected_layer].original_mask.clone();
                        let (width, height) = mask.dimensions();

                        // Paint the brush stroke
                        for dy in -(radius as i32)..=(radius as i32) {
                            for dx in -(radius as i32)..=(radius as i32) {
                                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                                if dist <= radius {
                                    let px = img_x + dx;
                                    let py = img_y + dy;

                                    if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                                        mask.put_pixel(px as u32, py as u32, color);
                                    }
                                }
                            }
                        }

                        // If we have a previous position, draw a line
                        if let Some(last_pos) = last_pos {
                            let last_rel_x = (last_pos.x - image_rect.min.x) / rect_size.x;
                            let last_rel_y = (last_pos.y - image_rect.min.y) / rect_size.y;
                            let last_img_x = (last_rel_x * image_size[0] as f32) as i32;
                            let last_img_y = (last_rel_y * image_size[1] as f32) as i32;

                            // Interpolate between points for smooth stroke
                            let dist = ((img_x - last_img_x).pow(2) + (img_y - last_img_y).pow(2)) as f32;
                            if dist > 0.0 {
                                let steps = (dist.sqrt() / (brush_size / 4.0)).max(1.0) as i32;
                                for i in 0..=steps {
                                    let t = i as f32 / steps as f32;
                                    let interp_x = last_img_x + ((img_x - last_img_x) as f32 * t) as i32;
                                    let interp_y = last_img_y + ((img_y - last_img_y) as f32 * t) as i32;

                                    for dy in -(radius as i32)..=(radius as i32) {
                                        for dx in -(radius as i32)..=(radius as i32) {
                                            let dist = ((dx * dx + dy * dy) as f32).sqrt();
                                            if dist <= radius {
                                                let px = interp_x + dx;
                                                let py = interp_y + dy;

                                                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                                                    mask.put_pixel(px as u32, py as u32, color);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Update the selection with the new mask
                        selections[selected_layer] = Selection::from_mask(ctx, mask);
                    }
                }));
            } else {
                canvas.brush_enabled = false;
                canvas.brush_paint_callback = None;
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
