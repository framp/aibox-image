use anyhow::Result;
use serde::{Serialize, de::DeserializeOwned};

pub mod types;
pub mod zmq;

pub trait Transport {
    async fn send<Req>(&self, req: Req) -> Result<Req::Response>
    where
        Req: IntoResponse + Serialize + Into<types::Request> + Send,
        Req::Response: DeserializeOwned;
}

pub trait IntoResponse {
    type Response;
}
