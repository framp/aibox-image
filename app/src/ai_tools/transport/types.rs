use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use super::IntoResponse;
use crate::config::{InpaintingModel, SelectionModel};

#[derive(Serialize, Debug)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Request {
    ImageSelection(ImageSelectionRequest),
    Inpaint(InpaintRequest),
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
#[serde(untagged)]
pub enum ModelKind {
    Selection(SelectionModel),
    Inpainting(InpaintingModel),
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
