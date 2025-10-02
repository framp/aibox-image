use eframe::egui::Ui;

use crate::{config::Config, image_canvas::ImageCanvas, worker::ErrorChan};

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
        let tools = vec![
            Box::new(selection::SelectionTool::new(config, tx_err.clone())) as Box<dyn Tool>,
            Box::new(inpaint::InpaintTool::new(config, tx_err.clone())) as Box<dyn Tool>,
            Box::new(upscale::UpscaleTool::new(config, tx_err.clone())) as Box<dyn Tool>,
            Box::new(portrait::PortraitTool::new(config, tx_err.clone())) as Box<dyn Tool>,
            Box::new(face_swap::FaceSwapTool::new(config, tx_err.clone())) as Box<dyn Tool>,
        ];

        Self { tools }
    }

    pub fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas, _has_image: bool) {
        ui.heading("🛠 Tools");

        for tool in self.tools.iter_mut() {
            ui.separator();
            ui.add_space(5.0);
            tool.show(ui, canvas);
        }
    }
}
