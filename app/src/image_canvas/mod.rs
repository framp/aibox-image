use eframe::egui::{Color32, ColorImage, TextureHandle, Vec2};
use fast_morphology::{
    BorderMode, ImageSize, KernelShape, MorphExOp, MorphScalar, MorphologyThreadingPolicy,
    morphology,
};
use image::{DynamicImage, GrayImage};

use std::path::PathBuf;

pub struct Image {
    texture: TextureHandle,
    image: DynamicImage,
    size: Vec2,
}

impl Image {
    pub fn new(ctx: &eframe::egui::Context, image: DynamicImage) -> Self {
        let rgba_image = image.to_rgba8();
        let (width, height) = rgba_image.dimensions();
        let color_image =
            ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba_image);
        let texture = ctx.load_texture("image", color_image, Default::default());

        Self {
            texture,
            image,
            size: Vec2::new(width as f32, height as f32),
        }
    }

    pub fn image(&self) -> &DynamicImage {
        &self.image
    }

    pub fn size(&self) -> Vec2 {
        self.size
    }
}

#[derive(Clone)]
pub struct Selection {
    pub mask_texture: TextureHandle,
    pub overlay_texture: TextureHandle,
    pub mask: GrayImage,
    pub original_mask: GrayImage,
    pub applied_mask: GrayImage,
    pub growth: i32,
    pub blur: u32,
    pub visible: bool,
}

impl Selection {
    pub fn from_mask(ctx: &eframe::egui::Context, mask: GrayImage) -> Self {
        let (width, height) = mask.dimensions();
        let default_growth = ((width + height) as f32 / 2.0 / 90.0).round() as i32;
        let default_blur = 10u32;

        let original_mask = mask.clone();
        let preview_mask = Self::apply_growth_only(&original_mask, default_growth);
        let applied_mask =
            Self::apply_transforms(&original_mask, default_growth, true, default_blur);

        // Convert preview mask to textures (only growth, no blur)
        let pixels = preview_mask
            .pixels()
            .flat_map(|p| {
                let mask_alpha = p[0];
                Color32::from_white_alpha(mask_alpha).to_array()
            })
            .collect::<Vec<_>>();

        let overlay_texture = ctx.load_texture(
            "",
            ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &pixels),
            Default::default(),
        );
        let mask_texture = ctx.load_texture(
            "",
            ColorImage::from_gray([width as usize, height as usize], preview_mask.as_raw()),
            Default::default(),
        );

        Self {
            mask_texture,
            overlay_texture,
            mask: preview_mask,
            original_mask,
            applied_mask,
            growth: default_growth,
            blur: default_blur,
            visible: true,
        }
    }

    pub fn update_applied_mask(&mut self, ctx: &eframe::egui::Context) {
        // Update preview mask (growth only, no blur for preview)
        let preview_mask = Self::apply_growth_only(&self.original_mask, self.growth);
        self.mask = preview_mask.clone();

        // Update applied mask (growth + blur for inpainting)
        self.applied_mask =
            Self::apply_transforms(&self.original_mask, self.growth, true, self.blur);

        let (width, height) = preview_mask.dimensions();

        // Update textures with preview mask (no blur)
        let pixels = preview_mask
            .pixels()
            .flat_map(|p| {
                let mask_alpha = p[0];
                Color32::from_white_alpha(mask_alpha).to_array()
            })
            .collect::<Vec<_>>();

        self.overlay_texture = ctx.load_texture(
            "",
            ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &pixels),
            Default::default(),
        );
        self.mask_texture = ctx.load_texture(
            "",
            ColorImage::from_gray([width as usize, height as usize], preview_mask.as_raw()),
            Default::default(),
        );
    }

    fn apply_transforms(mask: &GrayImage, growth: i32, apply_blur: bool, blur: u32) -> GrayImage {
        let mut result = mask.clone();

        if growth != 0 {
            let abs_growth = growth.abs() as u32;
            if abs_growth > 0 {
                let kernel_size = abs_growth * 2 + 1; // Ensure odd kernel size
                let (width, height) = result.dimensions();

                // Create a disk structuring element
                let se_size = kernel_size as usize;
                let mut structuring_element = vec![0u8; se_size * se_size];
                let radius = se_size as f32 / 2.0;
                let center = se_size / 2;

                for y in 0..se_size {
                    for x in 0..se_size {
                        let dx = x as f32 - center as f32;
                        let dy = y as f32 - center as f32;
                        if (dx * dx + dy * dy).sqrt() <= radius {
                            structuring_element[y * se_size + x] = 255;
                        }
                    }
                }

                let mut dst = vec![0u8; (width * height) as usize];

                let op = if growth > 0 {
                    MorphExOp::Dilate
                } else {
                    MorphExOp::Erode
                };

                morphology(
                    result.as_raw(),
                    &mut dst,
                    op,
                    ImageSize::new(width as usize, height as usize),
                    &structuring_element,
                    KernelShape::new(se_size, se_size),
                    BorderMode::default(),
                    MorphScalar::default(),
                    MorphologyThreadingPolicy::default(),
                )
                .unwrap();

                result = GrayImage::from_raw(width, height, dst).unwrap();
            }
        }

        // Apply fast blur only if requested (for inpainting, not preview)
        if apply_blur && blur > 0 {
            let sigma = blur as f32 / 3.0;
            let dynamic_result = DynamicImage::ImageLuma8(result);
            result = dynamic_result.fast_blur(sigma).into_luma8();
        }

        result
    }

    // Helper function for preview (growth only, no blur)
    fn apply_growth_only(mask: &GrayImage, growth: i32) -> GrayImage {
        Self::apply_transforms(mask, growth, false, 0)
    }

    pub fn overlay(&self, ui: &mut eframe::egui::Ui, rect: eframe::egui::Rect) {
        if self.visible {
            ui.painter().image(
                self.overlay_texture.id(),
                rect,
                eframe::egui::Rect::from_min_max(
                    eframe::egui::pos2(0.0, 0.0),
                    eframe::egui::pos2(1.0, 1.0),
                ),
                eframe::egui::Color32::from_rgba_unmultiplied(128, 0, 128, 128),
            );
        }
    }
}

pub struct ImageCanvas {
    original_image: Option<DynamicImage>,
    pub image: Option<Image>,
    pub zoom: f32,
    pub offset: Vec2,
    pub is_dragging: bool,
    pub drag_start: eframe::egui::Pos2,
    pub selections: Vec<Selection>,
    pub brush_enabled: bool,
    pub brush_callback: Option<
        Box<
            dyn FnMut(eframe::egui::Pos2, Option<eframe::egui::Pos2>, eframe::egui::Rect) + 'static,
        >,
    >,
    pub brush_paint_callback: Option<
        Box<
            dyn FnMut(&mut Vec<Selection>, eframe::egui::Pos2, Option<eframe::egui::Pos2>, eframe::egui::Rect, &eframe::egui::Context) + 'static,
        >,
    >,
}

impl Default for ImageCanvas {
    fn default() -> Self {
        Self {
            original_image: None,
            image: None,
            zoom: 1.0,
            offset: Vec2::ZERO,
            is_dragging: false,
            drag_start: eframe::egui::Pos2::ZERO,
            selections: Vec::new(),
            brush_enabled: false,
            brush_callback: None,
            brush_paint_callback: None,
        }
    }
}

impl ImageCanvas {
    pub fn open_file_dialog(&mut self, ctx: &eframe::egui::Context) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Image Files",
                &["png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif"],
            )
            .pick_file()
        {
            let _ = self.load_image(path, ctx);
        }
    }

    pub fn has_image(&self) -> bool {
        self.image.is_some()
    }

    pub fn reset_zoom(&mut self) {
        self.zoom = 1.0;
        self.offset = Vec2::ZERO;
    }

    pub fn set_image(&mut self, image: DynamicImage, ctx: &eframe::egui::Context) {
        self.image = Some(Image::new(ctx, image));
        self.reset_zoom();
    }

    pub fn clear_image(&mut self, ctx: &eframe::egui::Context) {
        self.image = self
            .original_image
            .as_ref()
            .and_then(|img| Some(Image::new(ctx, img.clone())));

        self.reset_zoom();
    }

    pub fn load_image(&mut self, path: PathBuf, ctx: &eframe::egui::Context) -> Result<(), String> {
        let image_result = image::open(&path);

        match image_result {
            Ok(dynamic_image) => {
                self.original_image = Some(dynamic_image.clone());
                self.set_image(dynamic_image, ctx);

                Ok(())
            }
            Err(e) => Err(format!("Failed to load image: {}", e)),
        }
    }

    pub fn show(&mut self, ui: &mut eframe::egui::Ui) {
        if let Some(image) = &self.image {
            let available_size = ui.available_size();
            let response =
                ui.allocate_response(available_size, eframe::egui::Sense::click_and_drag());

            let scaled_size = image.size * self.zoom;
            let center = available_size * 0.5;
            let image_pos = center - scaled_size * 0.5 + self.offset;
            let image_rect =
                eframe::egui::Rect::from_min_size(response.rect.min + image_pos, scaled_size);

            if response.hovered() {
                ui.ctx().input(|i| {
                    let scroll_delta = i.smooth_scroll_delta.y;
                    if scroll_delta != 0.0 {
                        let zoom_factor = 1.0 + scroll_delta * 0.001;
                        self.zoom = (self.zoom * zoom_factor).clamp(0.1, 5.0);
                    }
                });
            }

            if self.brush_enabled && (self.brush_callback.is_some() || self.brush_paint_callback.is_some()) {
                // Brush mode: paint on the mask
                if response.dragged() {
                    if let Some(pointer_pos) = response.interact_pointer_pos() {
                        let last_pos = if !self.is_dragging {
                            self.is_dragging = true;
                            None
                        } else {
                            Some(self.drag_start)
                        };

                        if let Some(callback) = &mut self.brush_callback {
                            callback(pointer_pos, last_pos, image_rect);
                        }

                        if let Some(callback) = &mut self.brush_paint_callback {
                            callback(&mut self.selections, pointer_pos, last_pos, image_rect, ui.ctx());
                        }

                        self.drag_start = pointer_pos;
                    }
                } else {
                    self.is_dragging = false;
                }
            } else {
                // Pan mode: move the canvas
                if response.dragged() {
                    if !self.is_dragging {
                        self.is_dragging = true;
                        self.drag_start = response.interact_pointer_pos().unwrap_or_default();
                    }
                    if let Some(pointer_pos) = response.interact_pointer_pos() {
                        let delta = pointer_pos - self.drag_start;
                        self.offset += delta;
                        self.drag_start = pointer_pos;
                    }
                } else {
                    self.is_dragging = false;
                }
            }

            ui.painter().image(
                image.texture.id(),
                image_rect,
                eframe::egui::Rect::from_min_max(
                    eframe::egui::pos2(0.0, 0.0),
                    eframe::egui::pos2(1.0, 1.0),
                ),
                eframe::egui::Color32::WHITE,
            );

            for selection in &self.selections {
                selection.overlay(ui, image_rect);
            }

            ui.painter().text(
                response.rect.left_top() + eframe::egui::vec2(10.0, 10.0),
                eframe::egui::Align2::LEFT_TOP,
                format!(
                    "Zoom: {:.1}x | Size: {}x{}",
                    self.zoom, image.size.x as u32, image.size.y as u32
                ),
                eframe::egui::FontId::monospace(12.0),
                eframe::egui::Color32::WHITE,
            );
        } else {
            ui.centered_and_justified(|ui| {
                if ui.button("📁 Open Image").clicked() {
                    self.open_file_dialog(ui.ctx());
                }
            });
        }
    }
}
