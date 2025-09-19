use rmp_serde::Serializer;
use serde::{Serialize, de::DeserializeOwned};

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
    fn send<Req>(&self, req: Req) -> Result<Req::Response, Box<dyn std::error::Error>>
    where
        Req: IntoResponse + Serialize + Into<Request>,
        Req::Response: DeserializeOwned,
    {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::REQ)?;

        socket.set_connect_timeout(1000)?;
        socket.set_sndtimeo(1000)?;
        socket.connect(&self.endpoint)?;

        let mut buf = Vec::new();
        let request = Request::from(req.into());
        request.serialize(&mut Serializer::new(&mut buf).with_struct_map())?;
        socket.send(buf, 0)?;

        let raw = socket.recv_bytes(0)?;
        let decoded = rmp_serde::from_slice::<Req::Response>(&raw)?;
        Ok(decoded)
    }
}
