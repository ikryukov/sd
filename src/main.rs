use std::fs::File;

use anyhow::Result;
use clap::Parser;
use cudarc::cudnn::{ConvForward, TensorDescriptor};
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

    let device = cudarc::driver::CudaDevice::new(0)?;
    let cudnn_handle = cudarc::cudnn::Cudnn::new(device.clone()).unwrap();
    let desc = cudarc::cudnn::Cudnn::create_conv2d::<f32>(
        &cudnn_handle,
        [1, 1],
        [1, 1],
        [1, 1],
        cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )
    .unwrap();
    let input = cudarc::cudnn::Cudnn::create_4d_tensor::<f32>(
        &cudnn_handle,
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [1, 1, 10, 10],
    )
    .unwrap();
    let filter = cudarc::cudnn::Cudnn::create_4d_filter(
        &cudnn_handle,
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [1, 1, 3, 3],
    )
    .unwrap();
    let output = cudarc::cudnn::Cudnn::create_4d_tensor::<f32>(
        &cudnn_handle,
        cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        [1, 1, 10, 10],
    )
    .unwrap();
    let conv = ConvForward {
        conv: &desc,
        x: &input,
        w: &filter,
        y: &output,
    };

    let inp = device.htod_copy(vec![1.0f32; 10 * 10])?;
    let mut out = device.alloc_zeros::<f32>(10 * 10)?;
    let filter_d = device.htod_copy(vec![1.0f32 / 9.0f32; 3 * 3]).unwrap();

    let mut out = device.alloc_zeros::<f32>(10 * 10)?;

    let workspace_size = conv
        .get_workspace_size(
            cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        )
        .unwrap();
    let mut workspace = device.alloc_zeros::<u8>(workspace_size).unwrap();
    unsafe {
        conv.launch(
            cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            Some(&mut workspace),
            (1.0f32, 0.0f32),
            &inp,
            &filter_d,
            &mut out,
        )
        .unwrap();
    }

    let host_out = device.dtoh_sync_copy(&out).unwrap();

    info!("result: {:?}", host_out);

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
