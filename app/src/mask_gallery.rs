use eframe::egui::{ComboBox, Image, TextureHandle, Ui};
use image::{GrayImage, Luma};

#[derive(Clone)]
pub struct MaskItem {
    pub id: usize,
    pub name: String,
    pub mask: GrayImage,
    pub texture: TextureHandle,
}

impl MaskItem {
    pub fn new(id: usize, name: String, mask: GrayImage, ctx: &eframe::egui::Context) -> Self {
        let texture = ctx.load_texture(
            format!("mask_{}", id),
            eframe::egui::ColorImage::from_gray(
                [mask.width() as usize, mask.height() as usize],
                mask.as_raw(),
            ),
            Default::default(),
        );

        Self {
            id,
            name,
            mask,
            texture,
        }
    }
}

#[derive(Default)]
pub struct MaskGallery {
    pub masks: Vec<MaskItem>,
    next_id: usize,
    selected_masks: Vec<usize>, // IDs of selected masks for merging
    merge_operation: MergeOperation,
    new_mask_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeOperation {
    Add,
    Subtract,
}

impl Default for MergeOperation {
    fn default() -> Self {
        Self::Add
    }
}

#[derive(Clone, Debug)]
pub struct DraggedMask {
    pub id: usize,
    pub mask: GrayImage,
}

impl MaskGallery {
    pub fn new() -> Self {
        Self {
            masks: Vec::new(),
            next_id: 0,
            selected_masks: Vec::new(),
            merge_operation: MergeOperation::Add,
            new_mask_name: String::new(),
        }
    }

    pub fn add_mask(
        &mut self,
        name: String,
        mask: GrayImage,
        ctx: &eframe::egui::Context,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let mask_item = MaskItem::new(id, name, mask, ctx);
        self.masks.push(mask_item);

        id
    }

    pub fn get_mask(&self, id: usize) -> Option<&MaskItem> {
        self.masks.iter().find(|m| m.id == id)
    }

    pub fn remove_mask(&mut self, id: usize) {
        self.masks.retain(|m| m.id != id);
        self.selected_masks.retain(|&mask_id| mask_id != id);
    }

    pub fn merge_selected_masks(&self) -> Option<GrayImage> {
        if self.selected_masks.len() < 2 {
            return None;
        }

        let first_mask_id = self.selected_masks[0];
        let first_mask = self.get_mask(first_mask_id)?;
        let mut result = first_mask.mask.clone();

        for &mask_id in &self.selected_masks[1..] {
            if let Some(mask_item) = self.get_mask(mask_id) {
                result = match self.merge_operation {
                    MergeOperation::Add => add_masks(&result, &mask_item.mask),
                    MergeOperation::Subtract => subtract_masks(&result, &mask_item.mask),
                };
            }
        }

        Some(result)
    }

    pub fn show(&mut self, ui: &mut Ui, ctx: &eframe::egui::Context) {
        ui.heading("🎭 Mask Gallery");

        // Merge controls
        if self.masks.len() >= 2 {
            ui.separator();
            ui.label("Merge Masks:");

            ui.horizontal(|ui| {
                ui.label("Operation:");
                ComboBox::from_id_salt("merge_operation")
                    .selected_text(format!("{:?}", self.merge_operation))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.merge_operation, MergeOperation::Add, "Add");
                        ui.selectable_value(
                            &mut self.merge_operation,
                            MergeOperation::Subtract,
                            "Subtract",
                        );
                    });
            });

            ui.horizontal(|ui| {
                ui.label("New mask name:");
                ui.text_edit_singleline(&mut self.new_mask_name);
            });

            let can_merge = self.selected_masks.len() >= 2 && !self.new_mask_name.trim().is_empty();
            ui.add_enabled_ui(can_merge, |ui| {
                if ui.button("Merge Selected").clicked() {
                    if let Some(merged_mask) = self.merge_selected_masks() {
                        self.add_mask(self.new_mask_name.clone(), merged_mask, ctx);
                        self.selected_masks.clear();
                        self.new_mask_name.clear();
                    }
                }
            });

            ui.separator();
        }

        // Mask list
        let mut to_remove: Option<usize> = None;

        for mask_item in &self.masks {
            ui.horizontal(|ui| {
                // Selection checkbox
                let mut selected = self.selected_masks.contains(&mask_item.id);
                if ui.checkbox(&mut selected, "").changed() {
                    if selected {
                        self.selected_masks.push(mask_item.id);
                    } else {
                        self.selected_masks.retain(|&id| id != mask_item.id);
                    }
                }

                // Thumbnail with drag and drop support
                let size = mask_item.texture.size_vec2();
                let max_side = 64.0;
                let scale = (max_side / size.x.max(size.y)).min(1.0);
                let thumb_size = size * scale;

                let response = ui.add(Image::new(&mask_item.texture).fit_to_exact_size(thumb_size));

                // Handle drag and drop
                if response.hovered() {
                    ui.ctx().set_cursor_icon(eframe::egui::CursorIcon::Grab);
                }

                // Start drag operation
                let dragged_mask = DraggedMask {
                    id: mask_item.id,
                    mask: mask_item.mask.clone(),
                };

                ui.dnd_drag_source(
                    eframe::egui::Id::new(format!("mask_{}", mask_item.id)),
                    dragged_mask,
                    |ui| {
                        ui.add(Image::new(&mask_item.texture).fit_to_exact_size(thumb_size));
                    },
                );

                // Name and controls
                ui.vertical(|ui| {
                    ui.label(&mask_item.name);
                    if ui.small_button("Remove").clicked() {
                        to_remove = Some(mask_item.id);
                    }
                });
            });
            ui.separator();
        }

        if let Some(id) = to_remove {
            self.remove_mask(id);
        }

        if self.masks.is_empty() {
            ui.label("No masks in gallery. Create selections to add masks.");
        }
    }
}

fn add_masks(mask1: &GrayImage, mask2: &GrayImage) -> GrayImage {
    let (width, height) = (
        mask1.width().max(mask2.width()),
        mask1.height().max(mask2.height()),
    );

    let mut result = GrayImage::new(width, height);

    for x in 0..width {
        for y in 0..height {
            let pixel1 = if x < mask1.width() && y < mask1.height() {
                mask1.get_pixel(x, y)[0]
            } else {
                0
            };

            let pixel2 = if x < mask2.width() && y < mask2.height() {
                mask2.get_pixel(x, y)[0]
            } else {
                0
            };

            // Addition with clamping
            let combined = ((pixel1 as u16 + pixel2 as u16).min(255)) as u8;
            result.put_pixel(x, y, Luma([combined]));
        }
    }

    result
}

fn subtract_masks(mask1: &GrayImage, mask2: &GrayImage) -> GrayImage {
    let (width, height) = (mask1.width(), mask1.height());
    let mut result = mask1.clone();

    for x in 0..width {
        for y in 0..height {
            if x < mask2.width() && y < mask2.height() {
                let pixel1 = mask1.get_pixel(x, y)[0] as i16;
                let pixel2 = mask2.get_pixel(x, y)[0] as i16;

                // Subtraction with clamping to 0
                let combined = (pixel1 - pixel2).max(0) as u8;
                result.put_pixel(x, y, Luma([combined]));
            }
        }
    }

    result
}
