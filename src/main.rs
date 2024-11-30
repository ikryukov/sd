use anyhow::Result;
use clap::Parser;
use cudarc::cudnn::{sys, ConvForward, FilterDescriptor, TensorDescriptor};
use layers::{conv::Conv2d, layer::Layer};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::time::{Duration, Instant};
use tensor::{FilterTensor, Tensor};
use resnet::ResnetBlock2D;

use tracing::info;

mod cmd;
mod layers;
mod tensor;
mod resnet;
mod Resnet;

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

    // let x_tensor = Tensor {
    //     desc: cudarc::cudnn::Cudnn::create_4d_tensor::<f32>(
    //         &cudnn_handle,
    //         cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    //         [1, 1, 10, 10],
    //     )
    //     .unwrap(),
    //     data: device.htod_copy(vec![1.0f32; 10 * 10]).unwrap(),
    // };
    // let filter_tensor = FilterTensor {
    //     desc: cudarc::cudnn::Cudnn::create_4d_filter(
    //         &cudnn_handle,
    //         cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    //         [1, 1, 3, 3],
    //     )
    //     .unwrap(),
    //     data: device.htod_copy(vec![1.0f32 / 9.0f32; 3 * 3]).unwrap(),
    // };
    // let mut y_tensor = Tensor {
    //     desc: cudarc::cudnn::Cudnn::create_4d_tensor::<f32>(
    //         &cudnn_handle,
    //         cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    //         [1, 1, 10, 10],
    //     )
    //     .unwrap(),
    //     data: device.alloc_zeros::<f32>(10 * 10).unwrap(),
    // };

    // let test_conv = Conv2d::new(cudnn_handle.clone(), 1, 1, 3, 3, 1, 1, 1);
    // info!("{:?}", test_conv);

    // let workspace_size = test_conv.get_workspace_size(&x_tensor, &filter_tensor, &y_tensor);
    // let mut workspace = device.alloc_zeros::<u8>(workspace_size).unwrap();

    // test_conv.forward(
    //     Some(&mut workspace),
    //     &x_tensor,
    //     &filter_tensor,
    //     &mut y_tensor,
    // );

    // let host_out = device.dtoh_sync_copy(&y_tensor.data).unwrap();
    // info!("{:?}", host_out);


    // let contents = fs::read_to_string(args.model).expect("Couldn't find or load that file.");

    // let model_index: ModelIndex = serde_json::from_str(contents.as_str()).unwrap();

    // println!("{:?}", model_index);

    let file = File::open(args.weights).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = match SafeTensors::deserialize(&buffer) {
        Ok(t) => t,
        Err(e) => {
            println!("{:?}", e);
            return Err(e.into());
        }
    };
    println!("list tensors:");
    for (tensor_name, tensor_view) in tensors.tensors() {
        println!(
            "{} \t\t {:?} \t {:?}",
            tensor_name,
            tensor_view.shape(),
            tensor_view.dtype()
        );
    }
    Ok(())
}
