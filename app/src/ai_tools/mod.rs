use eframe::egui::Ui;

use crate::image_canvas::SharedCanvas;

mod selection;
mod zmq;

pub trait Tool {
    fn show(&mut self, ui: &mut Ui);
}

#[derive(Default)]
pub struct ToolsPanel {
    pub tools: Vec<Box<dyn Tool>>,
}

impl ToolsPanel {
    pub fn new(canvas: SharedCanvas) -> Self {
        let mut tools = Vec::new();

        tools.push(Box::new(selection::SelectionTool::new(canvas)) as Box<dyn Tool>);

        Self { tools }
    }

    pub fn show(&mut self, ui: &mut Ui, has_image: bool) {
        ui.heading("🛠 Tools");

        for (i, tool) in self.tools.iter_mut().enumerate() {
            ui.separator();
            ui.add_space(5.0);
            tool.show(ui);
        }
    }
}
