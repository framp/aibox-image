use std::error::Error;

use rmp_serde::Serializer;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_bytes::ByteBuf;

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Request {
    ImageSelection {
        prompt: String,
        image_path: String,
    },
    Inpaint {
        prompt: String,
        image_path: String,
        mask: ByteBuf,
    },
    HealthCheck,
    Shutdown,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Response<T> {
    Success(T),
    Error {
        message: String,
    },
    Healthy {
        service: String,
        sam_model: String,
        sam_device: String,
        gd_model: String,
        gd_device: String,
        timestamp: f64,
    },
    ShuttingDown {
        message: String,
    },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct ImageSelectionPayload {
    pub masks: Vec<ByteBuf>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct InpaintPayload {
    pub image: ByteBuf,
}

pub fn request_response<T>(endpoint: &str, request: Request) -> Result<Response<T>, Box<dyn Error>>
where
    T: DeserializeOwned,
{
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::REQ)?;

    socket.set_connect_timeout(1000)?;
    socket.set_sndtimeo(1000)?;
    // socket.set_rcvtimeo(30000)?;

    socket.connect(endpoint)?;

    let mut buf = Vec::new();
    let mut serializer = Serializer::new(&mut buf).with_struct_map();
    request.serialize(&mut serializer)?;

    socket.send(buf, 0)?;
    let raw = socket.recv_bytes(0)?;

    let decoded = rmp_serde::from_slice::<Response<T>>(&raw)?;

    Ok(decoded)
}
