use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use super::IntoResponse;
use crate::config::{InpaintingModel, PortraitEditingModel, SelectionModel, UpscalingModel};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExpressionParams {
    pub rotate_pitch: f64,
    pub rotate_yaw: f64,
    pub rotate_roll: f64,
    pub blink: f64,
    pub eyebrow: f64,
    pub wink: f64,
    pub pupil_x: f64,
    pub pupil_y: f64,
    pub aaa: f64,
    pub eee: f64,
    pub woo: f64,
    pub smile: f64,
    pub src_weight: f64,
}

impl Default for ExpressionParams {
    fn default() -> Self {
        Self {
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
            src_weight: 0.0,
        }
    }
}

#[derive(Serialize, Debug)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Request {
    ImageSelection(ImageSelectionRequest),
    Inpaint(InpaintRequest),
    Upscale(UpscaleRequest),
    EditExpression(EditExpressionRequest),
    Load(LoadRequest),
}

#[derive(Serialize, Debug)]
pub struct ImageSelectionRequest {
    pub prompt: String,
    pub image_bytes: ByteBuf,
    pub threshold: f32,
}

impl IntoResponse for ImageSelectionRequest {
    type Response = ImageSelectionResponse;
}

impl Into<Request> for ImageSelectionRequest {
    fn into(self) -> Request {
        Request::ImageSelection(self)
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct ImageSelectionResponse {
    pub masks: Vec<ByteBuf>,
}

#[derive(Serialize, Debug)]
pub struct InpaintRequest {
    pub prompt: String,
    pub image_bytes: ByteBuf,
    pub mask: ByteBuf,
}

impl IntoResponse for InpaintRequest {
    type Response = InpaintResponse;
}

impl Into<Request> for InpaintRequest {
    fn into(self) -> Request {
        Request::Inpaint(self)
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct InpaintResponse {
    pub image: ByteBuf,
}

#[derive(Serialize, Debug)]
pub struct UpscaleRequest {
    pub prompt: String,
    pub image_bytes: ByteBuf,
}

impl IntoResponse for UpscaleRequest {
    type Response = UpscaleResponse;
}

impl Into<Request> for UpscaleRequest {
    fn into(self) -> Request {
        Request::Upscale(self)
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct UpscaleResponse {
    pub image: ByteBuf,
}

#[derive(Serialize, Debug)]
pub struct EditExpressionRequest {
    pub image_bytes: ByteBuf,
    pub rotate_pitch: f64,
    pub rotate_yaw: f64,
    pub rotate_roll: f64,
    pub blink: f64,
    pub eyebrow: f64,
    pub wink: f64,
    pub pupil_x: f64,
    pub pupil_y: f64,
    pub aaa: f64,
    pub eee: f64,
    pub woo: f64,
    pub smile: f64,
    pub src_weight: f64,
}

impl IntoResponse for EditExpressionRequest {
    type Response = EditExpressionResponse;
}

impl Into<Request> for EditExpressionRequest {
    fn into(self) -> Request {
        Request::EditExpression(self)
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct EditExpressionResponse {
    pub image: ByteBuf,
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum ModelKind {
    Selection(SelectionModel),
    Inpainting(InpaintingModel),
    Upscaling(UpscalingModel),
    PortraitEditing(PortraitEditingModel),
}

#[derive(Serialize, Debug)]
pub struct LoadRequest {
    #[serde(flatten)]
    pub model: ModelKind,
    pub cache_dir: String,
}

impl IntoResponse for LoadRequest {
    type Response = EmptyResponse;
}

impl Into<Request> for LoadRequest {
    fn into(self) -> Request {
        Request::Load(self)
    }
}

#[derive(Deserialize, Debug)]
pub struct EmptyResponse {}
