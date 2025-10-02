use eframe::egui::{Button, ComboBox, DragValue, Slider, Ui};
use image::{GrayImage, Luma};

use crate::image_canvas::{ImageCanvas, Selection};

pub struct BrushTool {
    brush_size: f32,
    brush_type: BrushType,
    show_settings_popup: bool,
    selected_layer: Option<usize>,
    undo_stack: Vec<UndoAction>,
    redo_stack: Vec<UndoAction>,
    is_painting: bool,
    last_paint_pos: Option<eframe::egui::Pos2>,
}

#[derive(Clone, Copy, PartialEq)]
enum BrushType {
    Add,
    Delete,
}

#[derive(Clone)]
struct UndoAction {
    layer_index: usize,
    previous_mask: GrayImage,
}

impl Default for BrushTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BrushTool {
    pub fn new() -> Self {
        Self {
            brush_size: 20.0,
            brush_type: BrushType::Add,
            show_settings_popup: false,
            selected_layer: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            is_painting: false,
            last_paint_pos: None,
        }
    }

    fn ui_brush_controls(&mut self, ui: &mut Ui, canvas: &ImageCanvas) {
        let enabled = canvas.image.is_some() && !canvas.selections.is_empty();

        ui.horizontal(|ui| {
            ui.label("Brush Type:");
            ui.radio_value(&mut self.brush_type, BrushType::Add, "✏ Add");
            ui.radio_value(&mut self.brush_type, BrushType::Delete, "🗑 Delete");
        });

        ui.horizontal(|ui| {
            ui.label("Brush Size:");
            if ui
                .add_enabled(enabled, Button::new("⚙"))
                .on_hover_text("Brush settings")
                .clicked()
            {
                self.show_settings_popup = !self.show_settings_popup;
            }
            ui.label(format!("{:.0}px", self.brush_size));
        });

        if self.show_settings_popup {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        enabled,
                        Slider::new(&mut self.brush_size, 1.0..=200.0).text("Size"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        enabled,
                        DragValue::new(&mut self.brush_size)
                            .speed(1.0)
                            .range(1.0..=200.0),
                    );
                });
            });
        }

        ui.horizontal(|ui| {
            let can_undo = !self.undo_stack.is_empty();
            let can_redo = !self.redo_stack.is_empty();

            if ui
                .add_enabled(enabled && can_undo, Button::new("↶ Undo"))
                .clicked()
            {
                self.undo(canvas, ui.ctx());
            }

            if ui
                .add_enabled(enabled && can_redo, Button::new("↷ Redo"))
                .clicked()
            {
                self.redo(canvas, ui.ctx());
            }
        });
    }

    fn ui_layer_selection(&mut self, ui: &mut Ui, canvas: &ImageCanvas) {
        let enabled = canvas.image.is_some() && !canvas.selections.is_empty();

        ui.horizontal(|ui| {
            ui.label("Edit Layer:");
            ui.add_enabled_ui(enabled, |ui| {
                ComboBox::from_id_salt("layer_selector")
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

    fn undo(&mut self, canvas: &ImageCanvas, ctx: &eframe::egui::Context) {
        if let Some(action) = self.undo_stack.pop() {
            if action.layer_index < canvas.selections.len() {
                let current_mask = canvas.selections[action.layer_index].original_mask.clone();
                self.redo_stack.push(UndoAction {
                    layer_index: action.layer_index,
                    previous_mask: current_mask,
                });

                // Will be applied in the show method
            }
        }
    }

    fn redo(&mut self, canvas: &ImageCanvas, ctx: &eframe::egui::Context) {
        if let Some(action) = self.redo_stack.pop() {
            if action.layer_index < canvas.selections.len() {
                let current_mask = canvas.selections[action.layer_index].original_mask.clone();
                self.undo_stack.push(UndoAction {
                    layer_index: action.layer_index,
                    previous_mask: current_mask,
                });

                // Will be applied in the show method
            }
        }
    }

    fn apply_brush_stroke(
        &self,
        mask: &mut GrayImage,
        pos: eframe::egui::Pos2,
        image_rect: eframe::egui::Rect,
        canvas: &ImageCanvas,
    ) {
        // Convert screen position to image coordinates
        let image_size = canvas.image.as_ref().unwrap().size();
        let rect_size = image_rect.size();

        let rel_x = (pos.x - image_rect.min.x) / rect_size.x;
        let rel_y = (pos.y - image_rect.min.y) / rect_size.y;

        if rel_x < 0.0 || rel_x > 1.0 || rel_y < 0.0 || rel_y > 1.0 {
            return;
        }

        let img_x = (rel_x * image_size.x) as i32;
        let img_y = (rel_y * image_size.y) as i32;

        let radius = self.brush_size / 2.0;
        let color = match self.brush_type {
            BrushType::Add => Luma([255u8]),
            BrushType::Delete => Luma([0u8]),
        };

        let (width, height) = mask.dimensions();

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
    }

    fn draw_brush_line(
        &self,
        mask: &mut GrayImage,
        from: eframe::egui::Pos2,
        to: eframe::egui::Pos2,
        image_rect: eframe::egui::Rect,
        canvas: &ImageCanvas,
    ) {
        let dist = from.distance(to);
        let steps = (dist / (self.brush_size / 4.0)).max(1.0) as i32;

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let pos = eframe::egui::Pos2 {
                x: from.x + (to.x - from.x) * t,
                y: from.y + (to.y - from.y) * t,
            };
            self.apply_brush_stroke(mask, pos, image_rect, canvas);
        }
    }
}

impl super::Tool for BrushTool {
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.push_id("brush", |ui| {
            ui.label("Brush Tool");

            self.ui_brush_controls(ui, canvas);
            self.ui_layer_selection(ui, canvas);

            // Apply undo/redo actions
            if let Some(action) = self.undo_stack.last() {
                if action.layer_index < canvas.selections.len() {
                    canvas.selections[action.layer_index] =
                        Selection::from_mask(ui.ctx(), action.previous_mask.clone());
                    self.undo_stack.pop();
                }
            }

            if let Some(action) = self.redo_stack.last() {
                if action.layer_index < canvas.selections.len() {
                    canvas.selections[action.layer_index] =
                        Selection::from_mask(ui.ctx(), action.previous_mask.clone());
                    self.redo_stack.pop();
                }
            }
        });
    }
}
