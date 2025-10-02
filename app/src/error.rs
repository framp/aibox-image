
use crate::ai_tools::error::AiToolError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("ai tool error: {0}")]
    AITool(#[from] AiToolError),
    #[error(transparent)]
    External(#[from] anyhow::Error),
}
