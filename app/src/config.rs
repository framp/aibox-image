use std::path::{Path, PathBuf};

use config::{Config as ConfigTrait, File, FileFormat};
use serde::{Deserialize, Deserializer, Serialize, de};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub models: Models,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Models {
    #[serde(deserialize_with = "deserialize_pathbuf_absolute")]
    pub cache_dir: PathBuf,
    pub selection: Vec<ModelEntry<SelectionModel>>,
    pub inpainting: Vec<ModelEntry<InpaintingModel>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelEntry<T> {
    pub name: String,
    #[serde(flatten)]
    pub kind: T,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SelectionModel {
    #[serde(rename = "grounding_dino_sam")]
    GroundingDINOSAM {
        grounding_dino_model: String,
        sam_model: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InpaintingModel {
    #[serde(rename = "stable_diffusion")]
    StableDiffusion { model_or_checkpoint: String },
}

pub fn load() -> Result<Config, Box<dyn std::error::Error>> {
    let config = ConfigTrait::builder()
        .add_source(File::with_name("config").format(FileFormat::Toml))
        .add_source(
            File::with_name("config.override")
                .format(FileFormat::Toml)
                .required(false),
        )
        .build()?
        .try_deserialize()?;

    Ok(config)
}

fn deserialize_pathbuf_absolute<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    let path = PathBuf::from(s);
    if path.is_relative() {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(de::Error::custom)
    } else {
        Ok(path)
    }
}
