#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(rustdoc::missing_crate_level_docs)]

use eframe::egui;

mod ai_tools;
mod image_canvas;

use ai_tools::ToolsPanel;
use image_canvas::ImageCanvas;

fn main() -> eframe::Result {
    env_logger::init();

    let options: eframe::NativeOptions = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "AI Image Editor",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::<ImageEditorApp>::default())
        }),
    )
}

struct ImageEditorApp {
    image_canvas: ImageCanvas,
    tools_panel: ToolsPanel,
    error_message: Option<String>,
}

impl Default for ImageEditorApp {
    fn default() -> Self {
        let canvas = ImageCanvas::default();
        let tools_panel = ToolsPanel::new();

        Self {
            image_canvas: canvas,
            tools_panel,
            error_message: None,
        }
    }
}

impl ImageEditorApp {
    fn load_image(&mut self, file_path: std::path::PathBuf, ctx: &egui::Context) {
        match self.image_canvas.load_image(file_path.clone(), ctx) {
            Ok(()) => {
                self.error_message = None;
                println!("Successfully loaded: {:?}", file_path);
            }
            Err(e) => {
                self.error_message = Some(e);
            }
        }
    }

    fn open_file_dialog(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Image Files",
                &["png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif"],
            )
            .pick_file()
        {
            self.load_image(path, ctx);
        }
    }

    fn save_image(&mut self) {
        if let Some(current_path) = &self.image_canvas.image_path {
            match self.save_to_path(current_path.clone()) {
                Ok(()) => {
                    self.error_message = None;
                    println!("Saved to: {:?}", current_path);
                }
                Err(e) => {
                    self.error_message = Some(format!("Save failed: {}", e));
                }
            }
        } else {
            self.save_as_image();
        }
    }

    fn save_as_image(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG Image", &["png"])
            .add_filter("JPEG Image", &["jpg", "jpeg"])
            .add_filter("BMP Image", &["bmp"])
            .add_filter("All Images", &["png", "jpg", "jpeg", "bmp"])
            .save_file()
        {
            match self.save_to_path(path.clone()) {
                Ok(()) => {
                    self.error_message = None;
                    self.image_canvas.image_path = Some(path.clone());
                    println!("Saved as: {:?}", path);
                }
                Err(e) => {
                    self.error_message = Some(format!("Save failed: {}", e));
                }
            }
        }
    }

    fn save_to_path(&self, path: std::path::PathBuf) -> Result<(), String> {
        if let Some(_texture) = &self.image_canvas.texture {
            if let Some(original_path) = &self.image_canvas.image_path {
                std::fs::copy(original_path, &path)
                    .map_err(|e| format!("Failed to save file: {}", e))?;
                Ok(())
            } else {
                Err("No original image data available".to_string())
            }
        } else {
            Err("No image loaded".to_string())
        }
    }
}

impl eframe::App for ImageEditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::Area::new(egui::Id::new("main_window_area"))
            .fixed_pos(egui::pos2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_min_size(ctx.screen_rect().size());
                let drop_frame = egui::Frame::default()
                    .fill(egui::Color32::TRANSPARENT)
                    .stroke(egui::Stroke::NONE)
                    .inner_margin(egui::Margin::ZERO);

                let (_response, dropped_payload) =
                    ui.dnd_drop_zone::<std::path::PathBuf, _>(drop_frame, |ui| {
                        ui.set_min_size(ctx.screen_rect().size());
                        egui::TopBottomPanel::top("menu_bar").show_inside(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.menu_button("File", |ui| {
                                    if ui.button("Open Image...").clicked() {
                                        self.open_file_dialog(ctx);
                                        ui.close();
                                    }
                                    if ui.button("Load Test Image").clicked() {
                                        self.load_image(
                                            std::path::PathBuf::from("ferris.png"),
                                            ctx,
                                        );
                                        ui.close();
                                    }
                                    ui.separator();
                                    if ui.button("Save").clicked() {
                                        self.save_image();
                                        ui.close();
                                    }
                                    if ui.button("Save As...").clicked() {
                                        self.save_as_image();
                                        ui.close();
                                    }
                                    ui.separator();
                                    if ui.button("Exit").clicked() {
                                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                                    }
                                });
                            });
                        });
                        egui::SidePanel::right("tools_panel")
                            .default_width(300.0)
                            .min_width(250.0)
                            .max_width(400.0)
                            .show_inside(ui, |ui| {
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    let has_image = self.image_canvas.has_image();
                                    self.tools_panel.show(ui, &mut self.image_canvas, has_image);
                                });
                            });
                        egui::CentralPanel::default().show_inside(ui, |ui| {
                            if let Some(ref error) = self.error_message {
                                ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
                                ui.separator();
                            }
                            self.image_canvas.show(ui);
                        });
                    });
                if let Some(dropped_path) = dropped_payload {
                    let path = (*dropped_path).clone();
                    if let Some(extension) = path.extension() {
                        let ext = extension.to_string_lossy().to_lowercase();
                        if matches!(
                            ext.as_str(),
                            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp" | "tiff" | "tif"
                        ) {
                            self.load_image(path, ctx);
                        } else {
                            self.error_message = Some(format!("Unsupported file format: {}", ext));
                        }
                    } else {
                        self.error_message = Some("File has no extension".to_string());
                    }
                }
            });
    }
}
