use eframe::egui;

use crate::image_canvas::ImageCanvas;

pub struct HistoryPanel;

impl HistoryPanel {
    pub fn new() -> Self {
        Self
    }

    pub fn show(&mut self, ui: &mut egui::Ui, canvas: &mut ImageCanvas) {
        const FIXED_HEIGHT: f32 = 100.0;

        egui::Frame::default()
            .fill(egui::Color32::from_rgb(40, 40, 45))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(60, 60, 65)))
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.set_height(FIXED_HEIGHT);

                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("History:").strong());

                        let can_undo = canvas.history.can_undo();
                        let can_redo = canvas.history.can_redo();

                        if ui
                            .add_enabled(can_undo, egui::Button::new("⬅ Undo"))
                            .clicked()
                        {
                            if let Some(entry) = canvas.history.undo() {
                                let entry_clone = entry.clone();
                                canvas.restore_from_history(&entry_clone, ui.ctx());
                            }
                        }

                        if ui
                            .add_enabled(can_redo, egui::Button::new("Redo ➡"))
                            .clicked()
                        {
                            if let Some(entry) = canvas.history.redo() {
                                let entry_clone = entry.clone();
                                canvas.restore_from_history(&entry_clone, ui.ctx());
                            }
                        }
                    });

                    ui.separator();

                    let entries: Vec<_> = canvas
                        .history
                        .entries()
                        .iter()
                        .map(|e| e.action.clone())
                        .collect();
                    let current_idx = canvas.history.current_index();

                    egui::ScrollArea::vertical()
                        .id_salt("history_scroll")
                        .show(ui, |ui| {
                            for (idx, action) in entries.iter().enumerate() {
                                let is_current = current_idx == Some(idx);
                                let is_future = current_idx.map_or(false, |ci| idx > ci);

                                let text = format!("{}. {}", idx + 1, action.description());
                                let mut btn_text = egui::RichText::new(&text);

                                if is_current {
                                    btn_text = btn_text
                                        .strong()
                                        .color(egui::Color32::from_rgb(100, 200, 255));
                                } else if is_future {
                                    btn_text =
                                        btn_text.color(egui::Color32::from_rgb(120, 120, 120));
                                }

                                if ui.button(btn_text).clicked() {
                                    if let Some(entry) = canvas.history.goto(idx) {
                                        let entry_clone = entry.clone();
                                        canvas.restore_from_history(&entry_clone, ui.ctx());
                                    }
                                }
                            }
                        });
                });
            });
    }
}
