use crate::mask_gallery::DraggedMask;
use eframe::egui::{DragValue, Ui};
use image::{DynamicImage, GrayImage, RgbaImage};
use std::path::PathBuf;

#[derive(Clone)]
pub struct Layer {
    pub id: usize,
    pub name: String,
    pub image: Option<DynamicImage>,
    pub texture: Option<eframe::egui::TextureHandle>,
    pub visible: bool,
    pub mask: Option<GrayImage>,
    pub opacity: f32,
}

impl Layer {
    pub fn new(id: usize, name: String) -> Self {
        Self {
            id,
            name,
            image: None,
            texture: None,
            visible: true,
            mask: None,
            opacity: 1.0,
        }
    }

    pub fn new_with_image(
        id: usize,
        name: String,
        image: DynamicImage,
        ctx: &eframe::egui::Context,
    ) -> Self {
        let texture = create_texture_from_image(&image, &format!("layer_{}", id), ctx);

        Self {
            id,
            name,
            image: Some(image),
            texture: Some(texture),
            visible: true,
            mask: None,
            opacity: 1.0,
        }
    }

    pub fn set_image(&mut self, image: DynamicImage, ctx: &eframe::egui::Context) {
        let texture = create_texture_from_image(&image, &format!("layer_{}", self.id), ctx);
        self.image = Some(image);
        self.texture = Some(texture);
    }

    pub fn apply_mask(&mut self, mask: GrayImage) {
        self.mask = Some(mask);
    }

    pub fn remove_mask(&mut self) {
        self.mask = None;
    }

    pub fn has_mask(&self) -> bool {
        self.mask.is_some()
    }
}

#[derive(Default)]
pub struct LayerSystem {
    layers: Vec<Layer>,
    next_id: usize,
    selected_layer: Option<usize>,
    base_image_size: Option<(u32, u32)>,
}

impl LayerSystem {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            next_id: 0,
            selected_layer: None,
            base_image_size: None,
        }
    }

    pub fn set_base_image(
        &mut self,
        image: DynamicImage,
        _path: Option<PathBuf>,
        ctx: &eframe::egui::Context,
    ) {
        self.base_image_size = Some((image.width(), image.height()));

        // Clear existing layers and create base layer
        self.layers.clear();
        let base_layer = Layer::new_with_image(0, "Base Layer".to_string(), image, ctx);
        self.layers.push(base_layer);
        self.next_id = 1;
        self.selected_layer = Some(0);
    }

    pub fn add_layer(&mut self, name: String, _ctx: &eframe::egui::Context) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let layer = Layer::new(id, name);
        self.layers.push(layer);
        self.selected_layer = Some(id);

        id
    }

    pub fn add_layer_with_image(
        &mut self,
        name: String,
        image: DynamicImage,
        ctx: &eframe::egui::Context,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let layer = Layer::new_with_image(id, name, image, ctx);
        self.layers.push(layer);
        self.selected_layer = Some(id);

        id
    }

    pub fn remove_layer(&mut self, id: usize) -> bool {
        if self.layers.len() <= 1 {
            return false; // Don't remove the last layer
        }

        if let Some(index) = self.layers.iter().position(|l| l.id == id) {
            self.layers.remove(index);

            // Update selected layer if necessary
            if self.selected_layer == Some(id) {
                self.selected_layer = self.layers.first().map(|l| l.id);
            }

            true
        } else {
            false
        }
    }

    pub fn get_layer_mut(&mut self, id: usize) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub fn get_layer(&self, id: usize) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn move_layer_up(&mut self, id: usize) -> bool {
        if let Some(index) = self.layers.iter().position(|l| l.id == id) {
            if index < self.layers.len() - 1 {
                self.layers.swap(index, index + 1);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn move_layer_down(&mut self, id: usize) -> bool {
        if let Some(index) = self.layers.iter().position(|l| l.id == id) {
            if index > 0 {
                self.layers.swap(index, index - 1);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn get_layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn get_selected_layer_id(&self) -> Option<usize> {
        self.selected_layer
    }

    pub fn select_layer(&mut self, id: usize) {
        if self.layers.iter().any(|l| l.id == id) {
            self.selected_layer = Some(id);
        }
    }

    pub fn composite_layers(
        &self,
        ctx: &eframe::egui::Context,
    ) -> Option<eframe::egui::TextureHandle> {
        if self.layers.is_empty() {
            return None;
        }

        let (width, height) = self.base_image_size?;
        let mut composite = RgbaImage::new(width, height);

        // Render layers from bottom to top (reverse order in the array)
        for layer in self.layers.iter().rev() {
            if !layer.visible || layer.image.is_none() {
                continue;
            }

            let layer_image = layer.image.as_ref()?.to_rgba8();

            for x in 0..width {
                for y in 0..height {
                    if x < layer_image.width() && y < layer_image.height() {
                        let layer_pixel = layer_image.get_pixel(x, y);

                        // Apply mask if present
                        let mask_alpha = if let Some(ref mask) = layer.mask {
                            if x < mask.width() && y < mask.height() {
                                mask.get_pixel(x, y)[0] as f32 / 255.0
                            } else {
                                0.0
                            }
                        } else {
                            1.0
                        };

                        // Apply layer opacity
                        let final_alpha =
                            (layer_pixel[3] as f32 / 255.0) * layer.opacity * mask_alpha;

                        if final_alpha > 0.0 {
                            let composite_pixel = composite.get_pixel_mut(x, y);

                            // Alpha blending
                            let src_alpha = final_alpha;
                            let dst_alpha = composite_pixel[3] as f32 / 255.0;
                            let out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha);

                            if out_alpha > 0.0 {
                                for i in 0..3 {
                                    let src = layer_pixel[i] as f32 * src_alpha;
                                    let dst =
                                        composite_pixel[i] as f32 * dst_alpha * (1.0 - src_alpha);
                                    composite_pixel[i] = ((src + dst) / out_alpha) as u8;
                                }
                                composite_pixel[3] = (out_alpha * 255.0) as u8;
                            }
                        }
                    }
                }
            }
        }

        // Create texture from composite
        let color_image = eframe::egui::ColorImage::from_rgba_unmultiplied(
            [width as usize, height as usize],
            composite.as_raw(),
        );

        Some(ctx.load_texture("composite", color_image, Default::default()))
    }

    pub fn show_layer_panel(
        &mut self,
        ui: &mut Ui,
        _ctx: &eframe::egui::Context,
    ) -> LayerOperation {
        ui.heading("🎨 Layers");

        let mut operation = LayerOperation::None;

        // Add layer button
        if ui.button("Add Layer").clicked() {
            operation = LayerOperation::AddLayer;
        }

        ui.separator();

        // Layer list (render from top to bottom for UI, but composite in reverse)
        let mut to_remove: Option<usize> = None;
        let mut to_move_up: Option<usize> = None;
        let mut to_move_down: Option<usize> = None;
        let mut visibility_updates: Vec<(usize, bool)> = Vec::new();
        let mut opacity_updates: Vec<(usize, f32)> = Vec::new();

        for (index, layer) in self.layers.iter().enumerate() {
            let is_selected = self.selected_layer == Some(layer.id);

            // Create a drop zone for each layer that can accept masks
            let (group_response, dropped_mask) =
                ui.dnd_drop_zone::<DraggedMask, _>(eframe::egui::Frame::group(ui.style()), |ui| {
                    if is_selected {
                        ui.visuals_mut().widgets.inactive.bg_fill = ui.visuals().selection.bg_fill;
                    }

                    ui.horizontal(|ui| {
                        // Select layer by clicking
                        let response = ui.selectable_label(is_selected, &layer.name);
                        if response.clicked() {
                            self.selected_layer = Some(layer.id);
                        }

                        // Visibility toggle
                        let mut visible = layer.visible;
                        if ui.checkbox(&mut visible, "").changed() {
                            visibility_updates.push((layer.id, visible));
                        }
                    });

                    ui.horizontal(|ui| {
                        // Opacity slider
                        ui.label("Opacity:");
                        let mut opacity = layer.opacity;
                        if ui
                            .add(DragValue::new(&mut opacity).range(0.0..=1.0).speed(0.01))
                            .changed()
                        {
                            opacity_updates.push((layer.id, opacity));
                        }
                    });

                    if layer.has_mask() {
                        ui.horizontal(|ui| {
                            ui.label("🎭 Masked");
                            if ui.small_button("Remove Mask").clicked() {
                                operation = LayerOperation::RemoveMask(layer.id);
                            }
                        });
                    }

                    ui.horizontal(|ui| {
                        // Move buttons
                        if index < self.layers.len() - 1 {
                            if ui.small_button("↑").clicked() {
                                to_move_up = Some(layer.id);
                            }
                        }

                        if index > 0 {
                            if ui.small_button("↓").clicked() {
                                to_move_down = Some(layer.id);
                            }
                        }

                        // Remove button (only if not the last layer)
                        if self.layers.len() > 1 {
                            if ui.small_button("Remove").clicked() {
                                to_remove = Some(layer.id);
                            }
                        }
                    });
                });

            // Handle mask drop
            if let Some(dragged_mask) = dropped_mask {
                operation = LayerOperation::ApplyMask(layer.id, dragged_mask.mask.clone());
            }

            // Show drop indicator when hovering during drag
            if group_response.response.hovered() && ui.input(|i| i.pointer.any_down()) {
                // Visual feedback for drop zone (simplified)
                ui.label("🎯 Drop mask here");
            }

            ui.add_space(4.0);
        }

        // Apply updates
        for (id, visible) in visibility_updates {
            if let Some(layer) = self.get_layer_mut(id) {
                layer.visible = visible;
            }
        }

        for (id, opacity) in opacity_updates {
            if let Some(layer) = self.get_layer_mut(id) {
                layer.opacity = opacity;
            }
        }

        // Handle operations
        if let Some(id) = to_remove {
            operation = LayerOperation::RemoveLayer(id);
        }

        if let Some(id) = to_move_up {
            self.move_layer_up(id);
        }

        if let Some(id) = to_move_down {
            self.move_layer_down(id);
        }

        operation
    }

    pub fn has_layers(&self) -> bool {
        !self.layers.is_empty()
    }

    pub fn restore_from_state(&mut self, layers: Vec<Layer>, selected_layer: Option<usize>) {
        self.layers = layers;
        self.selected_layer = selected_layer;
    }

    pub fn get_composite_image(&self) -> Option<RgbaImage> {
        if self.layers.is_empty() {
            return None;
        }

        let (width, height) = self.base_image_size?;
        let mut composite = RgbaImage::new(width, height);

        // Render layers from bottom to top (reverse order in the array)
        for layer in self.layers.iter().rev() {
            if !layer.visible || layer.image.is_none() {
                continue;
            }

            let layer_image = layer.image.as_ref()?.to_rgba8();

            for x in 0..width {
                for y in 0..height {
                    if x < layer_image.width() && y < layer_image.height() {
                        let layer_pixel = layer_image.get_pixel(x, y);

                        // Apply mask if present
                        let mask_alpha = if let Some(ref mask) = layer.mask {
                            if x < mask.width() && y < mask.height() {
                                mask.get_pixel(x, y)[0] as f32 / 255.0
                            } else {
                                0.0
                            }
                        } else {
                            1.0
                        };

                        // Apply layer opacity
                        let final_alpha =
                            (layer_pixel[3] as f32 / 255.0) * layer.opacity * mask_alpha;

                        if final_alpha > 0.0 {
                            let composite_pixel = composite.get_pixel_mut(x, y);

                            // Alpha blending
                            let src_alpha = final_alpha;
                            let dst_alpha = composite_pixel[3] as f32 / 255.0;
                            let out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha);

                            if out_alpha > 0.0 {
                                for i in 0..3 {
                                    let src = layer_pixel[i] as f32 * src_alpha;
                                    let dst =
                                        composite_pixel[i] as f32 * dst_alpha * (1.0 - src_alpha);
                                    composite_pixel[i] = ((src + dst) / out_alpha) as u8;
                                }
                                composite_pixel[3] = (out_alpha * 255.0) as u8;
                            }
                        }
                    }
                }
            }
        }

        Some(composite)
    }
}

#[derive(Debug)]
pub enum LayerOperation {
    None,
    AddLayer,
    AddLayerWithImage(String, DynamicImage),
    RemoveLayer(usize),
    ApplyMask(usize, GrayImage),
    RemoveMask(usize),
}

fn create_texture_from_image(
    image: &DynamicImage,
    name: &str,
    ctx: &eframe::egui::Context,
) -> eframe::egui::TextureHandle {
    let rgba_image = image.to_rgba8();
    let (width, height) = rgba_image.dimensions();

    let color_image = eframe::egui::ColorImage::from_rgba_unmultiplied(
        [width as usize, height as usize],
        &rgba_image,
    );

    ctx.load_texture(name, color_image, Default::default())
}
