use anyhow::Result;
use clap::Parser;
use layers::layer::Layer;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::time::{Duration, Instant};
use tensor::{FilterTensor, Tensor};

// Platform-specific imports
#[cfg(not(target_os = "macos"))]
use cudarc::cudnn::{sys, ConvForward, FilterDescriptor, TensorDescriptor};
#[cfg(not(target_os = "macos"))]
use layers::conv::Conv2d;
#[cfg(not(target_os = "macos"))]
use resnet::ResnetBlock2D;

#[cfg(target_os = "macos")]
use metal::Device;
#[cfg(target_os = "macos")]
use layers::metal_conv::{MetalConv2d, ConvImplementation};
#[cfg(target_os = "macos")]
use layers::metal_conv_mps::MetalConv2dMPS;

use tracing::info;

mod cmd;
mod layers;
mod tensor;
#[cfg(not(target_os = "macos"))]
mod resnet;
#[cfg(not(target_os = "macos"))]
mod Resnet;
mod examples;

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

    let args = cmd::Args::parse();

    match &args.command {
        #[cfg(target_os = "macos")]
        Some(cmd::Commands::MetalConv) => {
            info!("Running Metal convolution example");
            examples::metal_conv_example();
            return Ok(());
        },
        #[cfg(not(target_os = "macos"))]
        Some(cmd::Commands::CudaConv) => {
            info!("Running CUDA convolution example");
            // This would call a CUDA example function
            // examples::cuda_conv_example();
            info!("CUDA convolution example not implemented yet");
            return Ok(());
        },
        Some(cmd::Commands::Benchmark { iterations, input_size }) => {
            info!("Running benchmark");
            run_benchmark(iterations.unwrap_or(10), input_size.unwrap_or(224))?;
            return Ok(());
        },
        Some(cmd::Commands::Run { model, weights }) => {
            run_model(model, weights)
        },
        None => {
            // For backward compatibility
            if let (Some(model), Some(weights)) = (args.model, args.weights) {
                run_model(&model, &weights)
            } else {
                info!("No command specified. Use --help for usage information.");
                Ok(())
            }
        }
    }
}

fn run_model(model_path: &std::path::Path, weights_path: &std::path::Path) -> Result<()> {
    info!("Running standard model");
    
    // Initialize platform-specific resources
    #[cfg(not(target_os = "macos"))]
    let device = cudarc::driver::CudaDevice::new(0)?;
    #[cfg(not(target_os = "macos"))]
    let cudnn_handle = cudarc::cudnn::Cudnn::new(device.clone()).unwrap();
    
    #[cfg(target_os = "macos")]
    let device = Device::system_default().expect("No Metal device found");
    #[cfg(target_os = "macos")]
    info!("Using Metal device: {}", device.name());

    // Load model and weights
    let file = File::open(model_path).unwrap();
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

    let file = File::open(weights_path).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = match SafeTensors::deserialize(&buffer) {
        Ok(t) => t,
        Err(e) => {
            println!("{:?}", e);
            return Err(e.into());
        }
    };
    
    // Platform-specific processing
    #[cfg(not(target_os = "macos"))]
    process_with_cuda(&device, &cudnn_handle, &tensors);
    
    #[cfg(target_os = "macos")]
    process_with_metal(&device, &tensors);

    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn process_with_cuda(device: &cudarc::driver::CudaDevice, cudnn_handle: &cudarc::cudnn::Cudnn, tensors: &SafeTensors) {
    // CUDA-specific processing code
    info!("Processing with CUDA");
    
    // Example: Create a simple convolution layer
    let in_channels = 3;
    let out_channels = 64;
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;
    let dilation = 1;
    
    // Create a CUDA convolution layer
    let conv = Conv2d::new(
        cudnn_handle.clone(),
        in_channels,
        out_channels,
        kernel_size,
        kernel_size,
        stride,
        padding,
        dilation,
    );
    info!("Created CUDA convolution layer");
    
    // Process tensors from the safetensors file
    for (tensor_name, tensor_view) in tensors.tensors() {
        info!(
            "Processing tensor: {} with shape {:?} and dtype {:?}",
            tensor_name,
            tensor_view.shape(),
            tensor_view.dtype()
        );
        
        // Here you would process each tensor based on its name and shape
        // For example, loading weights into the appropriate layers
    }
}

#[cfg(target_os = "macos")]
fn process_with_metal(device: &Device, tensors: &SafeTensors) {
    // Metal-specific processing code
    info!("Processing with Metal");
    
    // Example: Create a simple convolution layer
    let in_channels = 3;
    let out_channels = 64;
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;
    let dilation = 1;
    
    // Create a basic Metal convolution layer
    let basic_conv = MetalConv2d::new(
        device.clone(),
        in_channels,
        out_channels,
        kernel_size,
        kernel_size,
        stride,
        padding,
        dilation,
        ConvImplementation::Basic,
    );
    
    // Create an optimized Metal convolution layer if the shader exists
    let optimized_shader_path = std::path::Path::new("metal-shaders/conv2d_optimized.metal");
    if optimized_shader_path.exists() {
        let optimized_conv = MetalConv2d::new(
            device.clone(),
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            stride,
            padding,
            dilation,
            ConvImplementation::Optimized,
        );
        info!("Created optimized Metal convolution layer");
    }
    
    // Create an MPS convolution layer
    let mps_conv = MetalConv2dMPS::new(
        device.clone(),
        in_channels,
        out_channels,
        kernel_size,
        kernel_size,
        stride,
        padding,
        dilation,
    );
    info!("Created MPS convolution layer");
    
    // Process tensors from the safetensors file
    for (tensor_name, tensor_view) in tensors.tensors() {
        info!(
            "Processing tensor: {} with shape {:?} and dtype {:?}",
            tensor_name,
            tensor_view.shape(),
            tensor_view.dtype()
        );
        
        // Here you would process each tensor based on its name and shape
        // For example, loading weights into the appropriate layers
    }
}

fn run_benchmark(iterations: usize, input_size: usize) -> Result<()> {
    info!("Running {} iterations with input size {}", iterations, input_size);
    
    #[cfg(target_os = "macos")]
    {
        // Run Metal benchmarks
        let device = Device::system_default().expect("No Metal device found");
        info!("Using Metal device: {}", device.name());
        
        // Define benchmark parameters
        let batch_size = 1;
        let in_channels = 3;
        let out_channels = 64;
        let kernel_size = 3;
        let stride = 1;
        let padding = 1;
        let dilation = 1;
        
        // Create input and filter tensors
        use crate::layers::metal_conv::{MetalTensor, MetalFilterTensor};
        
        let input_size_total = batch_size * in_channels * input_size * input_size;
        let input_data = vec![1.0f32; input_size_total];
        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const _,
            (input_size_total * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input = MetalTensor {
            data: input_buffer,
            shape: vec![batch_size, in_channels, input_size, input_size],
        };
        
        let filter_size = out_channels * in_channels * kernel_size * kernel_size;
        let filter_data = vec![0.1f32; filter_size];
        let filter_buffer = device.new_buffer_with_data(
            filter_data.as_ptr() as *const _,
            (filter_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let filter = MetalFilterTensor {
            data: filter_buffer,
            shape: vec![out_channels, in_channels, kernel_size, kernel_size],
        };
        
        // Benchmark basic implementation
        let basic_conv = MetalConv2d::new(
            device.clone(),
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            stride,
            padding,
            dilation,
            ConvImplementation::Basic,
        );
        let mut basic_output = basic_conv.create_output_tensor(batch_size, input_size, input_size);
        
        info!("Benchmarking basic Metal implementation...");
        let start = Instant::now();
        for _ in 0..iterations {
            basic_conv.forward(&input, &filter, &mut basic_output);
        }
        let basic_duration = start.elapsed();
        let basic_avg = basic_duration.div_f64(iterations as f64);
        info!("Basic Metal implementation: {:?} total, {:?} avg per iteration", basic_duration, basic_avg);
        
        // Check if optimized shader exists
        let optimized_shader_path = std::path::Path::new("metal-shaders/conv2d_optimized.metal");
        if optimized_shader_path.exists() {
            // Benchmark optimized implementation
            let optimized_conv = MetalConv2d::new(
                device.clone(),
                in_channels,
                out_channels,
                kernel_size,
                kernel_size,
                stride,
                padding,
                dilation,
                ConvImplementation::Optimized,
            );
            let mut optimized_output = optimized_conv.create_output_tensor(batch_size, input_size, input_size);
            
            info!("Benchmarking optimized Metal implementation...");
            let start = Instant::now();
            for _ in 0..iterations {
                optimized_conv.forward(&input, &filter, &mut optimized_output);
            }
            let optimized_duration = start.elapsed();
            let optimized_avg = optimized_duration.div_f64(iterations as f64);
            info!("Optimized Metal implementation: {:?} total, {:?} avg per iteration", optimized_duration, optimized_avg);
            
            let speedup = basic_avg.as_secs_f64() / optimized_avg.as_secs_f64();
            info!("Optimized implementation is {:.2}x faster than basic", speedup);
        }
        
        // Benchmark MPS implementation
        let mps_conv = MetalConv2dMPS::new(
            device.clone(),
            in_channels,
            out_channels,
            kernel_size,
            kernel_size,
            stride,
            padding,
            dilation,
        );
        mps_conv.update_weights(&filter);
        let mut mps_output = mps_conv.create_output_tensor(batch_size, input_size, input_size);
        
        info!("Benchmarking MPS implementation...");
        let start = Instant::now();
        for _ in 0..iterations {
            mps_conv.forward(&input, &mut mps_output);
        }
        let mps_duration = start.elapsed();
        let mps_avg = mps_duration.div_f64(iterations as f64);
        info!("MPS implementation: {:?} total, {:?} avg per iteration", mps_duration, mps_avg);
        
        let mps_speedup = basic_avg.as_secs_f64() / mps_avg.as_secs_f64();
        info!("MPS implementation is {:.2}x faster than basic", mps_speedup);
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        // Run CUDA benchmarks
        let device = cudarc::driver::CudaDevice::new(0)?;
        let device_name = device.get_device_name()?;
        info!("Using CUDA device: {}", device_name);
        
        let cudnn_handle = cudarc::cudnn::Cudnn::new(device.clone()).unwrap();
        
        // Define benchmark parameters
        let batch_size = 1;
        let in_channels = 3;
        let out_channels = 64;
        let input_size = input_size; // Use the input size from the function parameter
        let kernel_size = 3;
        let stride = 1;
        let padding = 1;
        let dilation = 1;
        
        // Create input tensor
        let input_desc = cudarc::cudnn::Cudnn::create_tensor_descriptor(
            &[batch_size, in_channels, input_size, input_size],
            &[1, 1, 1, 1], // Strides - assuming contiguous memory
        ).unwrap();
        let input_data = vec![1.0f32; batch_size * in_channels * input_size * input_size];
        let input_cuda = device.htod_copy(input_data).unwrap();
        let input = Tensor {
            desc: input_desc,
            data: input_cuda,
        };
        
        // Create filter tensor
        let filter_desc = cudarc::cudnn::Cudnn::create_filter_descriptor(
            &[out_channels, in_channels, kernel_size, kernel_size],
        ).unwrap();
        let filter_data = vec![0.1f32; out_channels * in_channels * kernel_size * kernel_size];
        let filter_cuda = device.htod_copy(filter_data).unwrap();
        let filter = FilterTensor {
            desc: filter_desc,
            data: filter_cuda,
        };
        
        // Create output tensor
        let output_size = ((input_size as i32 + 2 * padding - dilation * (kernel_size as i32 - 1) - 1) / stride) + 1;
        let output_desc = cudarc::cudnn::Cudnn::create_tensor_descriptor(
            &[batch_size, out_channels, output_size as usize, output_size as usize],
            &[1, 1, 1, 1], // Strides - assuming contiguous memory
        ).unwrap();
        let output_cuda = device.alloc_zeros::<f32>(batch_size * out_channels * (output_size as usize) * (output_size as usize)).unwrap();
        let mut output = Tensor {
            desc: output_desc,
            data: output_cuda,
        };
        
        // Create Conv2d layer
        let conv = Conv2d::new(cudnn_handle.clone(), in_channels, out_channels, kernel_size, kernel_size, stride, padding, dilation);
        
        // Get workspace size
        let workspace_size = conv.get_workspace_size(&input, &filter, &output);
        let mut workspace = device.alloc_zeros::<u8>(workspace_size).unwrap();
        
        // Benchmark CUDA implementation
        info!("Benchmarking CUDA implementation...");
        let start = Instant::now();
        for _ in 0..iterations {
            conv.forward(Some(&mut workspace), &input, &filter, &mut output);
        }
        let cuda_duration = start.elapsed();
        let cuda_avg = cuda_duration.div_f64(iterations as f64);
        info!("CUDA implementation: {:?} total, {:?} avg per iteration", cuda_duration, cuda_avg);
    }
    
    Ok(())
}
