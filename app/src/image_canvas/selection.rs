use eframe::egui::{self, Color32, ColorImage, TextureHandle};
use image::GrayImage;

#[derive(Clone)]
pub struct Selection {
    pub mask_texture: TextureHandle,
    pub overlay_texture: TextureHandle,
    pub mask: GrayImage,
    pub visible: bool,
    pub scale: f32,
}

impl Selection {
    pub fn from_mask(ctx: &eframe::egui::Context, mask: GrayImage) -> Self {
        let mut sel = Self {
            mask_texture: ctx.load_texture("", ColorImage::example(), Default::default()),
            overlay_texture: ctx.load_texture("", ColorImage::example(), Default::default()),
            mask,
            visible: true,
            scale: 1.0,
        };
        sel.update_textures(ctx);
        sel
    }

    pub fn resize_mask(&mut self, ctx: &egui::Context, new_width: u32, new_height: u32) {
        let mut new_mask = GrayImage::from_pixel(new_width, new_height, image::Luma([0u8]));

        let x_offset = (new_width.saturating_sub(self.mask.width())) / 2;
        let y_offset = (new_height.saturating_sub(self.mask.height())) / 2;

        for y in 0..self.mask.height().min(new_height) {
            for x in 0..self.mask.width().min(new_width) {
                let px = self.mask.get_pixel(x, y);
                new_mask.put_pixel(x + x_offset, y + y_offset, *px);
            }
        }

        self.mask = new_mask;
        self.scale = 1.0;
        self.update_textures(ctx);
    }

    fn update_textures(&mut self, ctx: &egui::Context) {
        let (width, height) = self.mask.dimensions();

        let pixels = self
            .mask
            .pixels()
            .flat_map(|p| egui::Color32::from_white_alpha(p[0]).to_array())
            .collect::<Vec<_>>();

        self.overlay_texture = ctx.load_texture(
            "overlay",
            ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &pixels),
            Default::default(),
        );

        self.mask_texture = ctx.load_texture(
            "mask",
            ColorImage::from_gray([width as usize, height as usize], self.mask.as_raw()),
            Default::default(),
        );
    }

    pub fn overlay(&self, ui: &mut eframe::egui::Ui, rect: eframe::egui::Rect) {
        if self.visible {
            let center = rect.center();
            let scaled_size = rect.size() * self.scale;
            let scaled_rect = eframe::egui::Rect::from_center_size(center, scaled_size);

            ui.painter().image(
                self.overlay_texture.id(),
                scaled_rect,
                eframe::egui::Rect::from_min_max(
                    eframe::egui::pos2(0.0, 0.0),
                    eframe::egui::pos2(1.0, 1.0),
                ),
                eframe::egui::Color32::from_rgba_unmultiplied(128, 0, 128, 128),
            );

            let stroke = eframe::egui::Stroke::new(2.0, eframe::egui::Color32::WHITE);
            ui.painter()
                .rect_stroke(scaled_rect, 0.0, stroke, eframe::egui::StrokeKind::Inside);
        }
    }
}
