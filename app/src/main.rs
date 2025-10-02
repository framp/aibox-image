#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(rustdoc::missing_crate_level_docs)]

use anyhow::{Context, anyhow};
use eframe::egui;

mod ai_tools;
mod config;
mod error;
mod history;
mod history_panel;
mod image_canvas;
mod msg_panel;
mod worker;

use ai_tools::ToolsPanel;
use history_panel::HistoryPanel;
use image_canvas::ImageCanvas;
use once_cell::sync::Lazy;
use tokio::runtime::Runtime;

use crate::msg_panel::MsgPanel;

static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new()
        .context("Failed to create Tokio runtime")
        .unwrap()
});

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let config =
        config::load().map_err(|e| anyhow::anyhow!("Failed to load application config: {}", e))?;

    // Spawn a thread that keeps the runtime alive
    std::thread::spawn(|| {
        RUNTIME.block_on(futures::future::pending::<()>());
    });

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
            Ok(Box::new(ImageEditorApp::new(&config)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run eframe application: {:?}", e))
}

struct ImageEditorApp {
    image_canvas: ImageCanvas,
    tools_panel: ToolsPanel,
    history_panel: HistoryPanel,
    msg_panel: MsgPanel,
}

impl ImageEditorApp {
    fn new(config: &config::Config) -> Self {
        let canvas = ImageCanvas::default();
        let msg_panel = MsgPanel::new();
        let tools_panel = ToolsPanel::new(config, msg_panel.tx_error.clone());
        let history_panel = HistoryPanel::new();

        Self {
            image_canvas: canvas,
            tools_panel,
            history_panel,
            msg_panel,
        }
    }

    fn load_image(&mut self, file_path: std::path::PathBuf, ctx: &egui::Context) {
        match self.image_canvas.load_image(file_path.clone(), ctx) {
            Ok(()) => {
                self.msg_panel.clear();
                println!("Successfully loaded: {file_path:?}");

                if let Some(image) = self.image_canvas.image.as_ref() {
                    let path_str = file_path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("image")
                        .to_string();
                    self.image_canvas.history.push(
                        crate::history::Action::LoadImage { path: path_str },
                        image.image().clone(),
                        self.image_canvas.selections.clone(),
                    );
                }
            }
            Err(e) => {
                let _ = self.msg_panel.tx_error.try_send(anyhow!(e).into());
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
        self.save_as_image();
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
                    self.msg_panel.clear();
                    println!("Saved as: {path:?}");
                }
                Err(e) => {
                    let _ = self
                        .msg_panel
                        .tx_error
                        .try_send(anyhow!("Save failed: {}", e).into());
                }
            }
        }
    }

    fn save_to_path(&self, path: std::path::PathBuf) -> Result<(), String> {
        if let Some(image) = &self.image_canvas.image {
            image
                .image()
                .save(&path)
                .map_err(|e| format!("Failed to save file: {e}"))?;
            Ok(())
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
                                        self.open_file_dialog(ui.ctx());
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
                                ui.add_enabled_ui(self.image_canvas.has_image(), |ui| {
                                    ui.menu_button("Image", |ui| {
                                        if ui.button("Reset Zoom").clicked() {
                                            self.image_canvas.reset_zoom();
                                            ui.close();
                                        }
                                        if ui.button("Revert to Original").clicked() {
                                            self.image_canvas.clear_image(ui.ctx());
                                            ui.close();
                                        }
                                    });
                                });
                                ui.add_enabled_ui(self.msg_panel.has_messages(), |ui| {
                                    if ui.button("Messages").clicked() {
                                        self.msg_panel.open_modal();
                                    }
                                    self.msg_panel.show_modal(ui);
                                })
                            });
                        });
                        egui::SidePanel::right("tools_panel")
                            .default_width(350.0)
                            .min_width(350.0)
                            .max_width(450.0)
                            .show_inside(ui, |ui| {
                                self.history_panel.show(ui, &mut self.image_canvas);

                                ui.heading("🛠 Tools");
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    let has_image = self.image_canvas.has_image();
                                    self.tools_panel.show(ui, &mut self.image_canvas, has_image);
                                });
                            });
                        egui::CentralPanel::default().show_inside(ui, |ui| {
                            self.msg_panel.show_last(ui);
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
                            let _ = self
                                .msg_panel
                                .tx_error
                                .try_send(anyhow!("Unsupported file format: {}", ext).into());
                        }
                    } else {
                        let _ = self
                            .msg_panel
                            .tx_error
                            .try_send(anyhow!("File has no extension").into());
                    }
                }
            });
    }
}
