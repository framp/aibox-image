use futures::compat::Future01CompatExt;
use rmp_serde::Serializer;
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;
use tokio_zmq::{Multipart, Req, prelude::*};

use super::{IntoResponse, Transport, TransportError, types::Request};

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
    async fn send<ReqType>(&self, req: ReqType) -> Result<ReqType::Response, TransportError>
    where
        ReqType: IntoResponse + Serialize + Into<Request> + Send,
        ReqType::Response: DeserializeOwned,
    {
        let context = Arc::new(zmq::Context::new());

        let socket_future = Req::builder(Arc::clone(&context))
            .connect(&self.endpoint)
            .build();

        let mut buf = Vec::new();
        let request = Request::from(req.into());
        request.serialize(&mut Serializer::new(&mut buf).with_struct_map())?;

        let socket =
            socket_future
                .compat()
                .await
                .map_err(|e| TransportError::SocketBuildError {
                    message: e.to_string(),
                })?;

        let multipart = Multipart::from(vec![zmq::Message::from(&buf[..])]);

        let socket =
            socket
                .send(multipart)
                .compat()
                .await
                .map_err(|e| TransportError::SendError {
                    message: e.to_string(),
                })?;
        let (response_multipart, _socket) =
            socket
                .recv()
                .compat()
                .await
                .map_err(|e| TransportError::ReceiveError {
                    message: e.to_string(),
                })?;

        let raw_data = response_multipart
            .get(0)
            .ok_or(TransportError::EmptyResponse)?
            .to_vec();
        let decoded = rmp_serde::from_slice::<ReqType::Response>(&raw_data)?;

        Ok(decoded)
    }
}
