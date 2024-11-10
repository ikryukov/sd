use std::fs::File;

use anyhow::Result;
use clap::Parser;
use layers::{conv::Conv2d, layer::Layer};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs;
use tracing::info;

mod cmd;
mod layers;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelIndex {
    #[serde(rename = "_class_name")]
    class_name: String,

    #[serde(rename = "_diffusers_version")]
    diffusers_version: String,

    #[serde(rename = "force_zeros_for_empty_prompt")]
    force_zeros_for_empty_prompt: bool,

    #[serde(rename = "add_watermarker")]
    add_watermarker: Option<serde_json::Value>,

    #[serde(rename = "scheduler")]
    scheduler: Vec<String>,

    #[serde(rename = "text_encoder")]
    text_encoder: Vec<String>,

    #[serde(rename = "text_encoder_2")]
    text_encoder_2: Vec<String>,

    #[serde(rename = "tokenizer")]
    tokenizer: Vec<String>,

    #[serde(rename = "tokenizer_2")]
    tokenizer_2: Vec<String>,

    #[serde(rename = "unet")]
    unet: Vec<String>,

    #[serde(rename = "vae")]
    vae: Vec<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = cmd::Args::try_parse()?;

    let test_conv = Conv2d::new(3, 3, 3, 3, 1, 1, 0);
    info!("{:?}", test_conv);
    test_conv.forward();

    // let contents = fs::read_to_string(args.model).expect("Couldn't find or load that file.");

    // let model_index: ModelIndex = serde_json::from_str(contents.as_str()).unwrap();

    // println!("{:?}", model_index);

    // let file = File::open(args.weights).unwrap();
    // let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    // let tensors = match SafeTensors::deserialize(&buffer) {
    //     Ok(t) => t,
    //     Err(e) => {
    //         println!("{:?}", e);
    //         return Err(e.into());
    //     }
    // };
    // for (tensor_name, tensor_view) in tensors.tensors() {
    //     println!(
    //         "{} \t\t {:?} \t {:?}",
    //         tensor_name,
    //         tensor_view.shape(),
    //         tensor_view.dtype()
    //     );
    // }
    Ok(())
}
