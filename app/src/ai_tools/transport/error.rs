use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransportError {
    #[error("Failed to serialize request: {0}")]
    SerializationError(#[from] rmp_serde::encode::Error),

    #[error("Failed to deserialize response: {0}")]
    DeserializationError(#[from] rmp_serde::decode::Error),

    #[error("ZMQ socket build error: {message}")]
    SocketBuildError { message: String },

    #[error("ZMQ send error: {message}")]
    SendError { message: String },

    #[error("ZMQ receive error: {message}")]
    ReceiveError { message: String },

    #[error("Empty response from server")]
    EmptyResponse,
}
