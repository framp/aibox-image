use serde::{Serialize, de::DeserializeOwned};

pub mod types;
pub mod zmq;

pub trait Transport {
    fn send<Req>(&self, req: Req) -> Result<Req::Response, Box<dyn std::error::Error>>
    where
        Req: IntoResponse + Serialize + Into<types::Request>,
        Req::Response: DeserializeOwned;
}

pub trait IntoResponse {
    type Response;
}
