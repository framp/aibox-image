use eframe::egui::Ui;

use crate::{
    image_canvas::ImageCanvas, layer_system::LayerOperation, mask_gallery::MaskGallery,
    undo_redo::UndoRedoManager,
};

mod inpaint;
mod selection;
mod zmq;

pub trait Tool {
    fn show(
        &mut self,
        ui: &mut Ui,
        canvas: &mut ImageCanvas,
        mask_gallery: &mut MaskGallery,
        undo_redo_manager: &mut UndoRedoManager,
    ) -> Option<LayerOperation>;
}

#[derive(Default)]
pub struct ToolsPanel {
    pub tools: Vec<Box<dyn Tool>>,
}

impl ToolsPanel {
    pub fn new() -> Self {
        let mut tools = Vec::new();

        tools.push(Box::new(selection::SelectionTool::new()) as Box<dyn Tool>);
        tools.push(Box::new(inpaint::InpaintTool::new()) as Box<dyn Tool>);

        Self { tools }
    }

    pub fn show(
        &mut self,
        ui: &mut Ui,
        canvas: &mut ImageCanvas,
        mask_gallery: &mut MaskGallery,
        undo_redo_manager: &mut UndoRedoManager,
        _has_image: bool,
    ) -> Vec<LayerOperation> {
        ui.heading("🛠 Tools");

        let mut layer_operations = Vec::new();

        for (_i, tool) in self.tools.iter_mut().enumerate() {
            ui.separator();
            ui.add_space(5.0);
            if let Some(operation) = tool.show(ui, canvas, mask_gallery, undo_redo_manager) {
                layer_operations.push(operation);
            }
        }

        layer_operations
    }
}
