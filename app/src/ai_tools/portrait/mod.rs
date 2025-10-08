use eframe::egui::{Button, CollapsingHeader, ComboBox, DragValue, Ui};

use crate::{
    ai_tools::transport::types::ExpressionParams, config::Config, image_canvas::ImageCanvas,
    worker::ErrorChan,
};

mod worker;

pub struct PortraitTool {
    loading: bool,
    worker: worker::Worker,

    config: Config,
    selected_model: Option<String>,
    loading_model: Option<String>,

    // Expression parameters
    rotate_pitch: f32,
    rotate_yaw: f32,
    rotate_roll: f32,
    blink: f32,
    eyebrow: f32,
    wink: f32,
    pupil_x: f32,
    pupil_y: f32,
    aaa: f32,
    eee: f32,
    woo: f32,
    smile: f32,
    src_weight: f32,
}

impl PortraitTool {
    pub fn new(config: &Config, tx_err: ErrorChan) -> Self {
        let mut tool = Self {
            loading: false,
            worker: worker::Worker::new(tx_err),
            config: config.clone(),
            selected_model: None,
            loading_model: None,

            rotate_pitch: 0.0,
            rotate_yaw: 0.0,
            rotate_roll: 0.0,
            blink: 0.0,
            eyebrow: 0.0,
            wink: 0.0,
            pupil_x: 0.0,
            pupil_y: 0.0,
            aaa: 0.0,
            eee: 0.0,
            woo: 0.0,
            smile: 0.0,
            src_weight: 1.0,
        };

        if let Some(first_model) = tool.config.models.portrait_editing.first() {
            tool.loading = true;
            tool.loading_model = Some(first_model.name.clone());
            tool.worker
                .load(first_model.kind.clone(), &tool.config.models.cache_dir);
        }

        tool
    }

    fn reset_parameters(&mut self) {
        self.rotate_pitch = 0.0;
        self.rotate_yaw = 0.0;
        self.rotate_roll = 0.0;
        self.blink = 0.0;
        self.eyebrow = 0.0;
        self.wink = 0.0;
        self.pupil_x = 0.0;
        self.pupil_y = 0.0;
        self.aaa = 0.0;
        self.eee = 0.0;
        self.woo = 0.0;
        self.smile = 0.0;
        self.src_weight = 1.0;
    }
}

impl super::Tool for PortraitTool {
    fn name(&self) -> &str {
        "Portrait"
    }

    fn show(&mut self, ui: &mut Ui, canvas: &mut ImageCanvas) {
        ui.push_id("portrait", |ui| {
            ui.label("🎭 Portrait Editor");

            let mut clicked_model_id = None;
            ComboBox::from_label("Model")
                .selected_text(self.selected_model.as_deref().unwrap_or("Select..."))
                .show_ui(ui, |ui| {
                    for (id, obj) in self.config.models.portrait_editing.iter().enumerate() {
                        if ui.selectable_label(false, &obj.name).clicked() {
                            clicked_model_id = Some(id);
                        }
                    }
                });

            if let Some(id) = clicked_model_id {
                self.loading = true;
                self.loading_model = Some(self.config.models.portrait_editing[id].name.clone());

                let model_kind = &self.config.models.portrait_editing[id].kind.clone();
                let cache_dir = self.config.models.cache_dir.clone();

                self.worker.load(model_kind.clone(), &cache_dir);
            }

            let can_submit =
                canvas.image.is_some() && self.selected_model.is_some() && !self.loading;

            ui.separator();

            CollapsingHeader::new("Head Rotation")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Pitch:");
                        ui.add(
                            DragValue::new(&mut self.rotate_pitch)
                                .speed(0.1)
                                .range(-50.0..=50.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Yaw:");
                        ui.add(
                            DragValue::new(&mut self.rotate_yaw)
                                .speed(0.1)
                                .range(-50.0..=50.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Roll:");
                        ui.add(
                            DragValue::new(&mut self.rotate_roll)
                                .speed(0.1)
                                .range(-50.0..=50.0),
                        );
                    });
                });

            CollapsingHeader::new("Eyes")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Blink:");
                        ui.add(
                            DragValue::new(&mut self.blink)
                                .speed(0.1)
                                .range(-30.0..=15.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Eyebrow:");
                        ui.add(
                            DragValue::new(&mut self.eyebrow)
                                .speed(0.1)
                                .range(-30.0..=25.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Wink:");
                        ui.add(
                            DragValue::new(&mut self.wink)
                                .speed(0.1)
                                .range(-10.0..=25.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Pupil X:");
                        ui.add(
                            DragValue::new(&mut self.pupil_x)
                                .speed(0.1)
                                .range(-15.0..=15.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Pupil Y:");
                        ui.add(
                            DragValue::new(&mut self.pupil_y)
                                .speed(0.1)
                                .range(-15.0..=15.0),
                        );
                    });
                });

            CollapsingHeader::new("Mouth")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("AAA:");
                        ui.add(
                            DragValue::new(&mut self.aaa)
                                .speed(0.5)
                                .range(-30.0..=120.0),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("EEE:");
                        ui.add(DragValue::new(&mut self.eee).speed(0.1).range(-20.0..=15.0));
                    });

                    ui.horizontal(|ui| {
                        ui.label("WOO:");
                        ui.add(DragValue::new(&mut self.woo).speed(0.1).range(-20.0..=15.0));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Smile:");
                        ui.add(
                            DragValue::new(&mut self.smile)
                                .speed(0.01)
                                .range(-0.3..=1.3),
                        );
                    });
                });

            CollapsingHeader::new("Advanced")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Source Weight:");
                        ui.add(
                            DragValue::new(&mut self.src_weight)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                    });
                });

            ui.separator();

            ui.horizontal(|ui| {
                let submit = ui.add_enabled(can_submit, Button::new("Apply Expression"));
                let reset = ui.add_enabled(!self.loading, Button::new("Reset"));

                if reset.clicked() {
                    self.reset_parameters();
                }

                if submit.clicked() && can_submit {
                    self.loading = true;

                    let expression_params = ExpressionParams {
                        rotate_pitch: self.rotate_pitch as f64,
                        rotate_yaw: self.rotate_yaw as f64,
                        rotate_roll: self.rotate_roll as f64,
                        blink: self.blink as f64,
                        eyebrow: self.eyebrow as f64,
                        wink: self.wink as f64,
                        pupil_x: self.pupil_x as f64,
                        pupil_y: self.pupil_y as f64,
                        aaa: self.aaa as f64,
                        eee: self.eee as f64,
                        woo: self.woo as f64,
                        smile: self.smile as f64,
                        src_weight: self.src_weight as f64,
                    };

                    self.worker
                        .edit_expression(canvas.image.as_ref().unwrap().image(), expression_params);
                }
            });

            if self.loading {
                ui.spinner();
            }

            if let Some(image) = self.worker.edited_image() {
                self.loading = false;

                canvas.set_image_with_history(
                    image,
                    ui.ctx(),
                    crate::history::Action::PortraitEdit,
                );
            }

            if self.worker.loaded() {
                self.loading = false;
                if let Some(model_name) = self.loading_model.take() {
                    self.selected_model = Some(model_name);
                }
            }
        });
    }
}
