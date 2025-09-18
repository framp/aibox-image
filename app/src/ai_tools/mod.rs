use eframe::egui::Ui;

use crate::{config::Config, image_canvas::ImageCanvas};

mod inpaint;
mod selection;
mod zmq;

pub trait Tool {
    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas);
}

#[derive(Default)]
pub struct ToolsPanel {
    pub tools: Vec<Box<dyn Tool>>,
}

impl ToolsPanel {
    pub fn new(config: &Config) -> Self {
        let mut tools = Vec::new();

        tools.push(Box::new(selection::SelectionTool::new(&config)) as Box<dyn Tool>);
        tools.push(Box::new(inpaint::InpaintTool::new(&config)) as Box<dyn Tool>);

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
