use eframe::egui::Ui;

#[derive(Debug, Clone, PartialEq)]
pub enum ToolType {
    Select,
    Brush,
    Fill,
    Eraser,
    Eyedropper,
}

#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub tool_type: ToolType,
    pub icon: String,
}

impl Tool {
    pub fn new(name: &str, tool_type: ToolType, icon: &str) -> Self {
        Self {
            name: name.to_string(),
            tool_type,
            icon: icon.to_string(),
        }
    }
}

#[derive(Default)]
pub struct ToolsPanel {
    pub tools: Vec<Tool>,
    pub selected_tool: Option<usize>,
    pub brush_size: f32,
    pub brush_opacity: f32,
    pub primary_color: [f32; 3],
    pub secondary_color: [f32; 3],
}

impl ToolsPanel {
    pub fn new() -> Self {
        let mut tools = Vec::new();
        
        tools.push(Tool::new("Select", ToolType::Select, "⬚"));
        tools.push(Tool::new("Brush", ToolType::Brush, "🖌"));
        tools.push(Tool::new("Fill", ToolType::Fill, "🪣"));
        tools.push(Tool::new("Eraser", ToolType::Eraser, "🧹"));
        tools.push(Tool::new("Eyedropper", ToolType::Eyedropper, "💧"));

        Self {
            tools,
            selected_tool: Some(0),
            brush_size: 10.0,
            brush_opacity: 1.0,
            primary_color: [0.0, 0.0, 0.0],
            secondary_color: [1.0, 1.0, 1.0],
        }
    }

    pub fn show(&mut self, ui: &mut Ui, has_image: bool) {
        ui.heading("🛠 Tools");
        ui.separator();
        ui.label("Select Tool:");
        ui.add_space(5.0);

        for (i, tool) in self.tools.iter().enumerate() {
            let is_selected = self.selected_tool == Some(i);
            let response = ui.selectable_label(is_selected, format!("{} {}", tool.icon, tool.name));
            
            if response.clicked() {
                self.selected_tool = Some(i);
            }
        }

        ui.add_space(10.0);
        ui.separator();
        ui.label("Colors:");
        ui.horizontal(|ui| {
            ui.label("Primary:");
            ui.color_edit_button_rgb(&mut self.primary_color);
        });
        ui.horizontal(|ui| {
            ui.label("Secondary:");
            ui.color_edit_button_rgb(&mut self.secondary_color);
        });

        ui.add_space(10.0);
        ui.separator();
        if let Some(selected_idx) = self.selected_tool {
            if let Some(selected_tool) = self.tools.get(selected_idx) {
                match selected_tool.tool_type {
                    ToolType::Brush | ToolType::Eraser => {
                        ui.label("Brush Settings:");
                        ui.horizontal(|ui| {
                            ui.label("Size:");
                            ui.add(eframe::egui::Slider::new(&mut self.brush_size, 1.0..=50.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Opacity:");
                            ui.add(eframe::egui::Slider::new(&mut self.brush_opacity, 0.0..=1.0));
                        });
                    }
                    ToolType::Fill => {
                        ui.label("Fill Settings:");
                        ui.horizontal(|ui| {
                            ui.label("Tolerance:");
                            let mut tolerance = 0.1f32;
                            ui.add(eframe::egui::Slider::new(&mut tolerance, 0.0..=1.0));
                        });
                    }
                    _ => {}
                }
            }
        }

        ui.add_space(10.0);
        ui.separator();
        ui.label("Actions:");
        ui.add_enabled_ui(has_image, |ui| {
            if ui.button("Clear Canvas").clicked() {
            }
            if ui.button("Undo").clicked() {
            }
            if ui.button("Redo").clicked() {
            }
        });

        if !has_image {
            ui.add_space(10.0);
            ui.colored_label(eframe::egui::Color32::GRAY, "Load an image to use tools");
        }
    }

    pub fn get_selected_tool(&self) -> Option<&Tool> {
        if let Some(idx) = self.selected_tool {
            self.tools.get(idx)
        } else {
            None
        }
    }
}