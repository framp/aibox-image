use eframe::egui::{Button, ColorImage, ComboBox, Image, TextureHandle, Ui};
use std::hash::{Hash, Hasher};
use tokio::sync::mpsc::Sender;

use crate::{config::Config, error::Error, image_canvas::ImageCanvas, worker::ErrorChan};

mod worker;

pub struct FaceSwapTool {
    loading: bool,
    analyzing_faces_source: bool,
    analyzing_faces_canvas: bool,
    worker: worker::Worker,

    config: Config,
    selected_model: Option<String>,
    source_image: Option<image::DynamicImage>,
    source_image_preview: Option<TextureHandle>,
    canvas_face_preview: Option<TextureHandle>,
    canvas_faces: Vec<crate::ai_tools::transport::types::FaceInfo>,
    selected_face_index: Option<i32>,
    last_canvas_image_hash: Option<u64>,
}

impl FaceSwapTool {
    pub fn new(config: &Config, tx_err: ErrorChan) -> Self {
        let mut tool = Self {
            loading: false,
            analyzing_faces_source: false,
            analyzing_faces_canvas: false,
            worker: worker::Worker::new(tx_err),
            config: config.clone(),
            selected_model: None,
            source_image: None,
            source_image_preview: None,
            canvas_face_preview: None,
            canvas_faces: Vec::new(),
            selected_face_index: None,
            last_canvas_image_hash: None,
        };

        if let Some(first_model) = tool.config.models.face_swapping.iter().next() {
            tool.selected_model = Some(first_model.name.clone());
            tool.loading = true;
            tool.worker
                .load(first_model.kind.clone(), &tool.config.models.cache_dir);
        }

        tool
    }
}

impl super::Tool for FaceSwapTool {
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.push_id("face_swap", |ui| {
            ui.label("Face Swap Tool");

            let mut clicked_model_id = None;
            ComboBox::from_label("Model")
                .selected_text(self.selected_model.as_deref().unwrap_or("Select..."))
                .show_ui(ui, |ui| {
                    for (id, obj) in self.config.models.face_swapping.iter().enumerate() {
                        if ui.selectable_label(false, &obj.name).clicked() {
                            clicked_model_id = Some(id);
                        }
                    }
                });

            if let Some(id) = clicked_model_id {
                self.loading = true;
                self.selected_model = Some(self.config.models.face_swapping[id].name.clone());

                let model_kind = &self.config.models.face_swapping[id].kind.clone();
                let cache_dir = self.config.models.cache_dir.clone();

                self.worker.load(model_kind.clone(), &cache_dir);
            }

            if ui.button("Load Source Image").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("image", &["png", "jpg", "jpeg"])
                    .pick_file()
                {
                    if let Ok(img) = image::open(path) {
                        self.source_image = Some(img.clone());

                        let rgba_image = img.to_rgba8();
                        let size = [rgba_image.width() as usize, rgba_image.height() as usize];
                        let color_image = ColorImage::from_rgba_unmultiplied(size, &rgba_image);
                        self.source_image_preview = Some(ui.ctx().load_texture(
                            "source_preview",
                            color_image,
                            Default::default(),
                        ));
                    }
                }
            }

            if let Some(source_texture) = &self.source_image_preview {
                ui.label("Source Image:");
                ui.add(
                    Image::from_texture(source_texture).max_size(source_texture.size_vec2() * 0.2),
                );
            }

            let mut should_analyze = false;
            if let Some(canvas_image) = canvas.image.as_ref() {
                let current_hash = {
                    let img = canvas_image.image();
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    (img.width(), img.height()).hash(&mut hasher);
                    hasher.finish()
                };

                if self.last_canvas_image_hash != Some(current_hash) {
                    self.last_canvas_image_hash = Some(current_hash);
                    self.canvas_faces.clear();
                    self.canvas_face_preview = None;
                    self.selected_face_index = None;
                    should_analyze = true;
                }
            }

            if self.selected_model.is_some()
                && canvas.image.is_some()
                && !self.analyzing_faces_canvas
                && (should_analyze || self.canvas_faces.is_empty())
            {
                self.analyzing_faces_canvas = true;
                self.worker
                    .analyze_faces(canvas.image.as_ref().unwrap().image());
            }

            if self.analyzing_faces_canvas {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Analyzing faces in image...");
                });
            }

            if let Some(canvas_texture) = &self.canvas_face_preview {
                ui.label(format!(
                    "Found {} face(s) in image:",
                    self.canvas_faces.len()
                ));
                ui.add(
                    Image::from_texture(canvas_texture).max_size(canvas_texture.size_vec2() * 0.3),
                );

                if !self.canvas_faces.is_empty() {
                    ui.label("Click on a face to select it for swapping:");

                    for (i, face) in self.canvas_faces.iter().enumerate() {
                        let is_selected = self.selected_face_index == Some(i as i32);
                        let button_text = if face.is_primary {
                            format!("Face {} (Primary)", i)
                        } else {
                            format!("Face {}", i)
                        };

                        if ui.selectable_label(is_selected, &button_text).clicked() {
                            self.selected_face_index = Some(i as i32);
                        }
                    }
                }
            }

            let can_submit = canvas.image.is_some()
                && self.source_image.is_some()
                && self.selected_model.is_some()
                && self.selected_face_index.is_some()
                && !self.loading
                && !self.analyzing_faces_canvas;

            let submit = ui.add_enabled(can_submit, Button::new("Swap Face"));
            let should_submit = submit.clicked();

            if should_submit && can_submit {
                self.loading = true;
                self.worker.swap_face(
                    self.source_image.as_ref().unwrap(),
                    canvas.image.as_ref().unwrap().image(),
                    self.selected_face_index.unwrap(),
                );
            }

            if self.loading {
                ui.spinner();
            }

            if let Some(faces_response) = self.worker.faces_analyzed() {
                self.analyzing_faces_canvas = false;
                self.canvas_faces = faces_response.faces;

                if !self.canvas_faces.is_empty() {
                    let preview_image =
                        image::load_from_memory(&faces_response.preview_image).unwrap();
                    let rgba_image = preview_image.to_rgba8();
                    let size = [rgba_image.width() as usize, rgba_image.height() as usize];
                    let color_image = ColorImage::from_rgba_unmultiplied(size, &rgba_image);
                    self.canvas_face_preview = Some(ui.ctx().load_texture(
                        "canvas_face_preview",
                        color_image,
                        Default::default(),
                    ));

                    if let Some(primary_face) = self.canvas_faces.iter().find(|f| f.is_primary) {
                        self.selected_face_index = Some(primary_face.index);
                    } else if !self.canvas_faces.is_empty() {
                        self.selected_face_index = Some(0);
                    }
                }
            }

            if let Some(image) = self.worker.swapped() {
                self.loading = false;
                canvas.set_image(image, ui.ctx());
            }

            if self.worker.loaded() {
                self.loading = false;

                if canvas.image.is_some()
                    && self.canvas_faces.is_empty()
                    && !self.analyzing_faces_canvas
                {
                    self.analyzing_faces_canvas = true;
                    self.worker
                        .analyze_faces(canvas.image.as_ref().unwrap().image());
                }
            }
        });
    }
}
