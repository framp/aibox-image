use serde::{Serialize, de::DeserializeOwned};

pub mod error;
pub mod types;
pub mod zmq;

pub use error::TransportError;

pub trait Transport {
    async fn send<Req>(&self, req: Req) -> Result<Req::Response, TransportError>
    where
        Req: IntoResponse + Serialize + Into<types::Request> + Send,
        Req::Response: DeserializeOwned;
}

pub trait IntoResponse {
    type Response;
}
