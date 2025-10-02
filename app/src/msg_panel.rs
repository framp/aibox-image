use std::time::{Duration, Instant};

use eframe::egui;

use crate::error::Error;

pub struct Message {
    kind: MessageKind,
    timestamp: Instant,
}

pub enum MessageKind {
    Error(Error),
}

pub struct MsgPanel {
    errors: Vec<Message>,
    rx_error: tokio::sync::mpsc::Receiver<Error>,
    pub tx_error: tokio::sync::mpsc::Sender<Error>,
    show_modal: bool,
}

const TOAST_DURATION: Duration = Duration::from_secs(5);

impl MsgPanel {
    pub fn new() -> Self {
        let (tx_error, rx_error) = tokio::sync::mpsc::channel(100);

        Self {
            errors: Vec::new(),
            rx_error,
            tx_error,
            show_modal: false,
        }
    }

    pub fn clear(&mut self) {
        self.errors.clear();
    }

    pub fn has_messages(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn open_modal(&mut self) {
        self.show_modal = true;
    }

    fn drain_queue(&mut self) {
        while let Ok(error) = self.rx_error.try_recv() {
            self.errors.push(Message {
                kind: MessageKind::Error(error),
                timestamp: Instant::now(),
            });
        }
    }

    fn show_msg(msg: &Message, ui: &mut egui::Ui) {
        let (color, text) = match &msg.kind {
            MessageKind::Error(e) => (egui::Color32::RED, format!("Error: {e}")),
        };

        ui.colored_label(color, text);
    }

    pub fn show_last(&mut self, ui: &mut egui::Ui) {
        self.drain_queue();

        if let Some(last) = self.errors.last() {
            if last.timestamp.elapsed() <= TOAST_DURATION {
                Self::show_msg(last, ui);
                ui.separator();
            }
        }
    }

    pub fn show_modal(&mut self, ui: &mut egui::Ui) {
        self.drain_queue();

        egui::Window::new("Error History")
            .collapsible(false)
            .resizable(true)
            .default_width(400.0)
            .default_height(300.0)
            .open(&mut self.show_modal)
            .show(ui.ctx(), |ui| {
                if ui.button("Clear All").clicked() {
                    self.errors.clear();
                }

                // Make scrollable
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut to_remove = Vec::new();

                    // Iterate all errors, newest last
                    for (i, error) in self.errors.iter().enumerate() {
                        ui.horizontal(|ui| {
                            Self::show_msg(error, ui);
                            if ui.button("❌").clicked() {
                                to_remove.push(i);
                            }
                        });
                        ui.separator();
                    }

                    // Remove marked errors in reverse order to keep indices valid
                    for idx in to_remove.into_iter().rev() {
                        self.errors.remove(idx);
                    }
                });
            });
    }
}
