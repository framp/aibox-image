use thiserror::Error;

use super::transport::TransportError;

#[derive(Error, Debug)]
pub enum WorkerError {
    #[error("Transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
