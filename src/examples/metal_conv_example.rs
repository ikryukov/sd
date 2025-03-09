use std::path::Path;
use std::time::Instant;

use metal::{Device, MTLResourceOptions};
use tracing::info;

use crate::layers::metal_conv::{MetalConv2d, MetalTensor, MetalFilterTensor, ConvImplementation};
use crate::layers::metal_conv_mps::MetalConv2dMPS;

/// Example demonstrating the use of the Metal Conv2d layer
pub fn metal_conv_example() {
    // Check if the shader files exist
    let basic_shader_path = Path::new("metal-shaders/conv2d.metal");
    let optimized_shader_path = Path::new("metal-shaders/conv2d_optimized.metal");
    
    if !basic_shader_path.exists() {
        panic!("Basic shader file not found: {}. Make sure you're running from the project root directory.", basic_shader_path.display());
    }
    
    if !optimized_shader_path.exists() {
        info!("Optimized shader file not found: {}. Only the basic implementation will be tested.", optimized_shader_path.display());
    }
    
    // Initialize Metal device
    let device = Device::system_default().expect("No Metal device found");
    info!("Using Metal device: {}", device.name());
    
    // Define layer parameters
    let batch_size = 1;
    let in_channels = 3;
    let out_channels = 64;
    let input_height = 224;
    let input_width = 224;
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;
    let dilation = 1;
    
    // Create input tensor (filled with 1.0)
    info!("Creating input tensor");
    let input_size = batch_size * in_channels * input_height * input_width;
    let input_data = vec![1.0f32; input_size];
    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const _,
        (input_size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input = MetalTensor {
        data: input_buffer,
        shape: vec![batch_size, in_channels, input_height, input_width],
    };
    
    // Create filter tensor (filled with 0.1)
    info!("Creating filter tensor");
    let filter_size = out_channels * in_channels * kernel_size * kernel_size;
    let filter_data = vec![0.1f32; filter_size];
    let filter_buffer = device.new_buffer_with_data(
        filter_data.as_ptr() as *const _,
        (filter_size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let filter = MetalFilterTensor {
        data: filter_buffer,
        shape: vec![out_channels, in_channels, kernel_size, kernel_size],
    };
    
    // Expected value: 1.0 * 0.1 * (3*3*3) = 2.7
    let expected = 0.1 * (kernel_size * kernel_size * in_channels) as f32;
    info!("Expected output value: {}", expected);
    
    // Test basic implementation
    info!("\n=== Testing Basic Implementation ===");
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
    
    let (output_height, output_width) = basic_conv.calculate_output_size(input_height, input_width);
    info!("Output dimensions: {}x{}", output_height, output_width);
    
    let mut basic_output = basic_conv.create_output_tensor(batch_size, input_height, input_width);
    
    let start = Instant::now();
    basic_conv.forward(&input, &filter, &mut basic_output);
    let basic_duration = start.elapsed();
    info!("Basic implementation took: {:?}", basic_duration);
    
    // Verify basic results
    let output_size = batch_size * out_channels * output_height * output_width;
    let mut basic_result = vec![0.0f32; output_size];
    let basic_output_ptr = basic_output.data.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(basic_output_ptr, basic_result.as_mut_ptr(), output_size);
    }
    
    info!("Basic first output value: {}", basic_result[0]);
    let basic_correct = basic_result.iter().all(|&x| (x - expected).abs() < 1e-5);
    info!("Basic all output values correct: {}", basic_correct);
    
    // Test optimized implementation if available
    let mut optimized_duration = basic_duration;  // Default to basic duration
    let mut optimized_speedup = 1.0;  // Default to no speedup
    
    if optimized_shader_path.exists() {
        info!("\n=== Testing Optimized Implementation ===");
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
        
        let mut optimized_output = optimized_conv.create_output_tensor(batch_size, input_height, input_width);
        
        let start = Instant::now();
        optimized_conv.forward(&input, &filter, &mut optimized_output);
        optimized_duration = start.elapsed();
        info!("Optimized implementation took: {:?}", optimized_duration);
        
        // Verify optimized results
        let mut optimized_result = vec![0.0f32; output_size];
        let optimized_output_ptr = optimized_output.data.contents() as *const f32;
        unsafe {
            std::ptr::copy_nonoverlapping(optimized_output_ptr, optimized_result.as_mut_ptr(), output_size);
        }
        
        info!("Optimized first output value: {}", optimized_result[0]);
        let optimized_correct = optimized_result.iter().all(|&x| (x - expected).abs() < 1e-5);
        info!("Optimized all output values correct: {}", optimized_correct);
        
        // Compare performance
        optimized_speedup = basic_duration.as_secs_f64() / optimized_duration.as_secs_f64();
        info!("Optimized implementation is {:.2}x faster than basic", optimized_speedup);
    }
    
    // Test MPS implementation
    info!("\n=== Testing MPS Implementation ===");
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
    
    // Update weights in the MPS convolution
    mps_conv.update_weights(&filter);
    
    let mut mps_output = mps_conv.create_output_tensor(batch_size, input_height, input_width);
    
    let start = Instant::now();
    mps_conv.forward(&input, &mut mps_output);
    let mps_duration = start.elapsed();
    info!("MPS implementation took: {:?}", mps_duration);
    
    // Verify MPS results
    let mut mps_result = vec![0.0f32; output_size];
    let mps_output_ptr = mps_output.data.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(mps_output_ptr, mps_result.as_mut_ptr(), output_size);
    }
    
    info!("MPS first output value: {}", mps_result[0]);
    let mps_correct = mps_result.iter().all(|&x| (x - expected).abs() < 1e-5);
    info!("MPS all output values correct: {}", mps_correct);
    
    // Compare performance with basic
    let mps_speedup = basic_duration.as_secs_f64() / mps_duration.as_secs_f64();
    info!("MPS implementation is {:.2}x faster than basic", mps_speedup);
    
    // Compare all implementations
    info!("\n=== Performance Summary ===");
    info!("Basic implementation: {:?}", basic_duration);
    
    if optimized_shader_path.exists() {
        info!("Optimized implementation: {:?} ({:.2}x faster than basic)", 
              optimized_duration, optimized_speedup);
        
        let mps_vs_optimized = optimized_duration.as_secs_f64() / mps_duration.as_secs_f64();
        info!("MPS implementation: {:?} ({:.2}x faster than basic, {:.2}x faster than optimized)", 
              mps_duration, mps_speedup, mps_vs_optimized);
    } else {
        info!("MPS implementation: {:?} ({:.2}x faster than basic)", 
              mps_duration, mps_speedup);
    }
} 