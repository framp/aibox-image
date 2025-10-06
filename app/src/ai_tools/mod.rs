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
    fn name(&self) -> &str;
}

pub struct ToolsPanel {
    pub tools: Vec<Box<dyn Tool>>,
    selected_tab: usize,
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

        Self {
            tools,
            selected_tab: 0,
        }
    }

    pub fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas, _has_image: bool) {
        ui.horizontal(|ui| {
            for (idx, tool) in self.tools.iter().enumerate() {
                if ui
                    .selectable_label(self.selected_tab == idx, tool.name())
                    .clicked()
                {
                    self.selected_tab = idx;
                }
            }
        });

        ui.separator();

        if let Some(tool) = self.tools.get_mut(self.selected_tab) {
            tool.show(ui, canvas);
        }
    }
}
