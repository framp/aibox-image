use eframe::egui::Ui;
use tokio::sync::mpsc::Sender;

use crate::{config::Config, error::Error, image_canvas::ImageCanvas, worker::ErrorChan};

pub mod error;
mod face_swap;
mod inpaint;
mod portrait;
mod selection;
mod transport;
mod upscale;

pub trait Tool {
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas);
}

#[derive(Default)]
pub struct ToolsPanel {
    pub tools: Vec<Box<dyn Tool>>,
}

impl ToolsPanel {
    pub fn new(config: &Config, tx_err: ErrorChan) -> Self {
        let mut tools = Vec::new();

        tools.push(
            Box::new(selection::SelectionTool::new(&config, tx_err.clone())) as Box<dyn Tool>,
        );
        tools.push(Box::new(inpaint::InpaintTool::new(&config, tx_err.clone())) as Box<dyn Tool>);
        tools.push(Box::new(upscale::UpscaleTool::new(&config, tx_err.clone())) as Box<dyn Tool>);
        tools.push(Box::new(portrait::PortraitTool::new(&config, tx_err.clone())) as Box<dyn Tool>);
        tools
            .push(Box::new(face_swap::FaceSwapTool::new(&config, tx_err.clone())) as Box<dyn Tool>);

        Self { tools }
    }

    pub fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas, _has_image: bool) {
        ui.heading("🛠 Tools");

        for (_i, tool) in self.tools.iter_mut().enumerate() {
            ui.separator();
            ui.add_space(5.0);
            tool.show(ui, canvas);
        }
    }
}
