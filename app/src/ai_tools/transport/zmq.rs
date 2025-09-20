use anyhow::{Context, Result};
use futures::compat::Future01CompatExt;
use rmp_serde::Serializer;
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;
use tokio_zmq::{Multipart, Req, prelude::*};

use super::{IntoResponse, Transport, types::Request};

#[derive(Clone)]
pub struct ZmqTransport {
    endpoint: String,
}

impl ZmqTransport {
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
        }
    }
}

impl Transport for ZmqTransport {
    async fn send<ReqType>(&self, req: ReqType) -> Result<ReqType::Response>
    where
        ReqType: IntoResponse + Serialize + Into<Request> + Send,
        ReqType::Response: DeserializeOwned,
    {
        let context = Arc::new(zmq::Context::new());

        // Build the socket using tokio-zmq's builder pattern
        let socket_future = Req::builder(Arc::clone(&context))
            .connect(&self.endpoint)
            .build();

        // Serialize the request
        let mut buf = Vec::new();
        let request = Request::from(req.into());
        request
            .serialize(&mut Serializer::new(&mut buf).with_struct_map())
            .context("Failed to serialize request")?;

        // Convert futures 0.1 to std::future using compat layer
        let socket = socket_future
            .compat()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to build ZMQ socket: {}", e))?;

        // Create multipart message
        let multipart = Multipart::from(vec![zmq::Message::from(&buf[..])]);

        // Send and receive using futures 0.1 -> std::future compat
        let socket = socket
            .send(multipart)
            .compat()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send ZMQ message: {}", e))?;
        let (response_multipart, _socket) = socket
            .recv()
            .compat()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to receive ZMQ response: {}", e))?;

        // Extract the response data
        let raw_data = response_multipart
            .get(0)
            .context("Empty response multipart")?
            .to_vec();
        let decoded = rmp_serde::from_slice::<ReqType::Response>(&raw_data)
            .context("Failed to deserialize response")?;

        Ok(decoded)
    }
}
