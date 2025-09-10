use eframe::egui::{ColorImage, ImageButton, TextureHandle, Vec2};
use image::{DynamicImage, ImageBuffer};
use std::{cell::RefCell, path::PathBuf, rc::Rc, str::Bytes};

pub type SharedCanvas = Rc<RefCell<ImageCanvas>>;

pub struct ImageCanvas {
    pub image_path: Option<PathBuf>,
    pub texture: Option<TextureHandle>,
    pub image_size: Vec2,
    pub zoom: f32,
    pub offset: Vec2,
    pub is_dragging: bool,
    pub drag_start: eframe::egui::Pos2,
    pub selections: Vec<DynamicImage>,
    pub selections_textures: Vec<TextureHandle>,
}

impl Default for ImageCanvas {
    fn default() -> Self {
        Self {
            image_path: None,
            texture: None,
            image_size: Vec2::ZERO,
            zoom: 1.0,
            offset: Vec2::ZERO,
            is_dragging: false,
            drag_start: eframe::egui::Pos2::ZERO,
            selections: Vec::new(),
            selections_textures: Vec::new(),
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

    pub fn set_image(
        &mut self,
        image: DynamicImage,
        path: Option<PathBuf>,
        ctx: &eframe::egui::Context,
    ) {
        let rgba_image = image.to_rgba8();
        let (width, height) = rgba_image.dimensions();

        let color_image =
            ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba_image);

        let texture = ctx.load_texture(
            format!(
                "image_{}",
                path.as_ref()
                    .map(|p| p.file_name().unwrap_or_default().to_string_lossy())
                    .unwrap_or("memory".into())
            ),
            color_image,
            Default::default(),
        );

        self.image_path = path;
        self.image_size = Vec2::new(width as f32, height as f32);
        self.texture = Some(texture);
        self.zoom = 1.0;
        self.offset = Vec2::ZERO;
    }

    pub fn load_image(&mut self, path: PathBuf, ctx: &eframe::egui::Context) -> Result<(), String> {
        let image_result = image::open(&path);

        match image_result {
            Ok(dynamic_image) => {
                self.set_image(dynamic_image, Some(path), ctx);

                Ok(())
            }
            Err(e) => Err(format!("Failed to load image: {}", e)),
        }
    }

    pub fn set_selections(&mut self, selections: Vec<DynamicImage>, ctx: &eframe::egui::Context) {
        self.selections_textures.clear();
        self.selections.clear();

        for (idx, dynamic_image) in selections.iter().enumerate() {
            let rgba_image = dynamic_image.to_rgba8();
            let (width, height) = rgba_image.dimensions();

            let color_image =
                ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba_image);

            let texture =
                ctx.load_texture(format!("image_{}", idx,), color_image, Default::default());

            self.selections_textures.push(texture);
            self.selections.push(dynamic_image.clone());
        }
    }

    pub fn show(&mut self, ui: &mut eframe::egui::Ui) {
        if let Some(texture) = &self.texture {
            let available_size = ui.available_size();
            let response =
                ui.allocate_response(available_size, eframe::egui::Sense::click_and_drag());
            if response.hovered() {
                ui.ctx().input(|i| {
                    let scroll_delta = i.smooth_scroll_delta.y;
                    if scroll_delta != 0.0 {
                        let zoom_factor = 1.0 + scroll_delta * 0.001;
                        self.zoom = (self.zoom * zoom_factor).clamp(0.1, 5.0);
                    }
                });
            }
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
            let scaled_size = self.image_size * self.zoom;
            let center = available_size * 0.5;
            let image_pos = center - scaled_size * 0.5 + self.offset;
            let image_rect =
                eframe::egui::Rect::from_min_size(response.rect.min + image_pos, scaled_size);

            ui.painter().image(
                texture.id(),
                image_rect,
                eframe::egui::Rect::from_min_max(
                    eframe::egui::pos2(0.0, 0.0),
                    eframe::egui::pos2(1.0, 1.0),
                ),
                eframe::egui::Color32::WHITE,
            );

            for texture in &self.selections_textures {
                ui.painter().image(
                    texture.id(),
                    image_rect,
                    eframe::egui::Rect::from_min_max(
                        eframe::egui::pos2(0.0, 0.0),
                        eframe::egui::pos2(1.0, 1.0),
                    ),
                    eframe::egui::Color32::WHITE,
                );
            }

            ui.painter().text(
                response.rect.left_top() + eframe::egui::vec2(10.0, 10.0),
                eframe::egui::Align2::LEFT_TOP,
                format!(
                    "Zoom: {:.1}x | Size: {}x{}",
                    self.zoom, self.image_size.x as u32, self.image_size.y as u32
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

    pub fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.offset = Vec2::ZERO;
    }

    pub fn has_image(&self) -> bool {
        self.texture.is_some()
    }

    pub fn clear_image(&mut self) {
        self.texture = None;
        self.image_path = None;
        self.image_size = Vec2::ZERO;
        self.zoom = 1.0;
        self.offset = Vec2::ZERO;
    }

    pub fn fit_to_window(&mut self) {
        if self.texture.is_some() {
            self.offset = Vec2::ZERO;
        }
    }

    pub fn reset_zoom(&mut self) {
        self.zoom = 1.0;
        self.offset = Vec2::ZERO;
    }
}
