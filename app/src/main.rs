#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(rustdoc::missing_crate_level_docs)]

use eframe::egui;

mod ai_tools;
mod image_canvas;
mod layer_system;
mod mask_gallery;
mod undo_redo;

use ai_tools::ToolsPanel;
use image_canvas::ImageCanvas;
use layer_system::{LayerOperation, LayerSystem};
use mask_gallery::MaskGallery;
use undo_redo::{LayerState, UndoRedoManager};

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
    undo_redo_manager: UndoRedoManager,
    mask_gallery: MaskGallery,
    layer_system: LayerSystem,
}

impl Default for ImageEditorApp {
    fn default() -> Self {
        let canvas = ImageCanvas::default();
        let tools_panel = ToolsPanel::new();

        Self {
            image_canvas: canvas,
            tools_panel,
            error_message: None,
            undo_redo_manager: UndoRedoManager::new(),
            mask_gallery: MaskGallery::new(),
            layer_system: LayerSystem::new(),
        }
    }
}

impl ImageEditorApp {
    fn save_layer_state_to_history(&mut self) {
        let state = LayerState::from_layer_system(&self.layer_system);
        self.undo_redo_manager.save_state(state);
    }

    fn perform_undo(&mut self, ctx: &egui::Context) {
        if let Some(state) = self.undo_redo_manager.undo() {
            self.layer_system
                .restore_from_state(state.layers.clone(), state.selected_layer);
            self.update_canvas_from_layers(ctx);
        }
    }

    fn perform_redo(&mut self, ctx: &egui::Context) {
        if let Some(state) = self.undo_redo_manager.redo() {
            self.layer_system
                .restore_from_state(state.layers.clone(), state.selected_layer);
            self.update_canvas_from_layers(ctx);
        }
    }

    fn update_canvas_from_layers(&mut self, ctx: &egui::Context) {
        if let Some(composite_texture) = self.layer_system.composite_layers(ctx) {
            self.image_canvas.texture = Some(composite_texture);
            if let Some(base_layer) = self.layer_system.get_layers().first() {
                if let Some(ref image) = base_layer.image {
                    self.image_canvas.image_size =
                        eframe::egui::Vec2::new(image.width() as f32, image.height() as f32);
                }
            }
        }
    }

    fn load_image(&mut self, file_path: std::path::PathBuf, ctx: &egui::Context) {
        match image::open(&file_path) {
            Ok(dynamic_image) => {
                // Clear all masks from the gallery
                self.mask_gallery.masks.clear();

                // Clear all selections from canvas
                self.image_canvas.selections.clear();

                // Set the base layer in the layer system (this already clears layers and creates new base layer)
                self.layer_system
                    .set_base_image(dynamic_image, Some(file_path.clone()), ctx);

                // Make sure canvas has the image path for AI tools to work
                self.image_canvas.image_path = Some(file_path.clone());

                self.save_layer_state_to_history();
                self.update_canvas_from_layers(ctx);
                self.error_message = None;
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load image: {}", e));
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
        if let Some(composite) = self.layer_system.get_composite_image() {
            // For now, save as PNG since we don't have a current path for layer composites
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG Image", &["png"])
                .set_file_name("composite.png")
                .save_file()
            {
                match composite.save(&path) {
                    Ok(()) => {
                        self.error_message = None;
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Save failed: {}", e));
                    }
                }
            }
        } else {
            self.error_message = Some("No layers to save".to_string());
        }
    }

    fn save_as_image(&mut self) {
        if let Some(composite) = self.layer_system.get_composite_image() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG Image", &["png"])
                .add_filter("JPEG Image", &["jpg", "jpeg"])
                .add_filter("BMP Image", &["bmp"])
                .add_filter("TIFF Image", &["tiff", "tif"])
                .save_file()
            {
                match composite.save(&path) {
                    Ok(()) => {
                        self.error_message = None;
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Save failed: {}", e));
                    }
                }
            }
        } else {
            self.error_message = Some("No layers to save".to_string());
        }
    }

    fn handle_layer_operation(&mut self, operation: LayerOperation, ctx: &egui::Context) {
        match operation {
            LayerOperation::AddLayer => {
                self.save_layer_state_to_history();
                let layer_name = format!("Layer {}", self.layer_system.get_layers().len() + 1);
                self.layer_system.add_layer(layer_name, ctx);
                self.update_canvas_from_layers(ctx);
            }
            LayerOperation::AddLayerWithImage(name, image) => {
                self.save_layer_state_to_history();
                self.layer_system.add_layer_with_image(name, image, ctx);
                self.update_canvas_from_layers(ctx);
            }
            LayerOperation::RemoveLayer(id) => {
                self.save_layer_state_to_history();
                self.layer_system.remove_layer(id);
                self.update_canvas_from_layers(ctx);
            }
            LayerOperation::ApplyMask(layer_id, mask) => {
                self.save_layer_state_to_history();
                if let Some(layer) = self.layer_system.get_layer_mut(layer_id) {
                    layer.apply_mask(mask);
                }
                self.update_canvas_from_layers(ctx);
            }
            LayerOperation::RemoveMask(layer_id) => {
                self.save_layer_state_to_history();
                if let Some(layer) = self.layer_system.get_layer_mut(layer_id) {
                    layer.remove_mask();
                }
                self.update_canvas_from_layers(ctx);
            }
            LayerOperation::None => {}
        }
    }

    fn handle_selection_to_mask(&mut self, ctx: &egui::Context) {
        // Convert current selection to mask and add to gallery
        if let Some(selection) = self.image_canvas.selections.first() {
            let mask_name = format!("Mask {}", self.mask_gallery.masks.len() + 1);
            self.mask_gallery
                .add_mask(mask_name, selection.mask.clone(), ctx);
        }
    }
}

impl eframe::App for ImageEditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle keyboard shortcuts
        ctx.input(|i| {
            if i.modifiers.ctrl {
                if i.key_pressed(egui::Key::Z) && !i.modifiers.shift {
                    self.perform_undo(ctx);
                } else if (i.key_pressed(egui::Key::Z) && i.modifiers.shift)
                    || i.key_pressed(egui::Key::Y)
                {
                    self.perform_redo(ctx);
                }
            }
        });

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

                                ui.separator();

                                // Undo/Redo buttons
                                ui.add_enabled_ui(self.undo_redo_manager.can_undo(), |ui| {
                                    if ui.button("↶ Undo").clicked() {
                                        self.perform_undo(ctx);
                                    }
                                });

                                ui.add_enabled_ui(self.undo_redo_manager.can_redo(), |ui| {
                                    if ui.button("↷ Redo").clicked() {
                                        self.perform_redo(ctx);
                                    }
                                });

                                ui.separator();

                                ui.label(format!(
                                    "History: {}/{}",
                                    self.undo_redo_manager.current_position(),
                                    self.undo_redo_manager.history_size()
                                ));

                                ui.separator();

                                // Selection to mask button
                                if !self.image_canvas.selections.is_empty() {
                                    if ui.button("Add Selection to Gallery").clicked() {
                                        self.handle_selection_to_mask(ctx);
                                    }
                                }
                            });
                        });

                        // Left sidebar for layers
                        egui::SidePanel::left("layer_panel")
                            .default_width(300.0)
                            .min_width(250.0)
                            .max_width(400.0)
                            .show_inside(ui, |ui| {
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    // Layer system
                                    let layer_op = self.layer_system.show_layer_panel(ui, ctx);
                                    self.handle_layer_operation(layer_op, ctx);

                                    ui.separator();
                                    ui.add_space(10.0);

                                    // Mask gallery
                                    self.mask_gallery.show(ui, ctx);
                                });
                            });

                        // Right sidebar for tools
                        egui::SidePanel::right("tools_panel")
                            .default_width(300.0)
                            .min_width(250.0)
                            .max_width(400.0)
                            .show_inside(ui, |ui| {
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    let has_image = self.layer_system.has_layers();
                                    let tool_operations = self.tools_panel.show(
                                        ui,
                                        &mut self.image_canvas,
                                        &mut self.mask_gallery,
                                        &mut self.undo_redo_manager,
                                        has_image,
                                    );

                                    // Handle layer operations from tools
                                    for operation in tool_operations {
                                        self.handle_layer_operation(operation, ctx);
                                    }
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
