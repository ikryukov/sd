use criterion::{criterion_group, criterion_main, Criterion};
use metal::{Device, MTLResourceOptions};
use std::time::Duration;

use sd::layers::metal_conv::{MetalConv2d, MetalTensor, MetalFilterTensor, ConvImplementation};
use sd::layers::metal_conv_mps::MetalConv2dMPS;

fn create_input_tensor(device: &Device, batch_size: usize, channels: usize, height: usize, width: usize) -> MetalTensor {
    let size = batch_size * channels * height * width;
    let data = vec![0.1f32; size];
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    
    MetalTensor {
        data: buffer,
        shape: vec![batch_size, channels, height, width],
    }
}

fn create_filter_tensor(device: &Device, out_channels: usize, in_channels: usize, kernel_size: usize) -> MetalFilterTensor {
    let size = out_channels * in_channels * kernel_size * kernel_size;
    let data = vec![0.1f32; size];
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    
    MetalFilterTensor {
        data: buffer,
        shape: vec![out_channels, in_channels, kernel_size, kernel_size],
    }
}

fn benchmark_metal_conv(c: &mut Criterion) {
    let device = Device::system_default().expect("No Metal device found");
    
    // Define test parameters with specific resolutions
    let batch_size = 1;
    let in_channels = 3;
    let out_channels = 64;
    // Define common resolutions: 800x600, 1024x768, and 1920x1080 (Full HD)
    let resolutions = [
        (800, 600),    // SVGA
        (1024, 768),   // XGA
        (1920, 1080),  // Full HD
    ];
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;
    let dilation = 1;
    
    // Create a benchmark group
    let mut group = c.benchmark_group("metal_convolution");
    group.measurement_time(Duration::from_secs(5)); // Reduced measurement time
    group.sample_size(10);
    
    for &(width, height) in &resolutions {
        // Create input and filter tensors
        let input = create_input_tensor(&device, batch_size, in_channels, height, width);
        let filter = create_filter_tensor(&device, out_channels, in_channels, kernel_size);
        
        // Basic implementation
        {
            let mut conv = MetalConv2d::new(
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
            
            let mut output = conv.create_output_tensor(batch_size, height, width);
            
            let id = format!("basic/{}x{}", width, height);
            group.bench_function(&id, |b| {
                b.iter(|| {
                    conv.forward(&input, &filter, &mut output);
                });
            });
        }
        
        // Optimized implementation
        {
            let mut conv = MetalConv2d::new(
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
            
            let mut output = conv.create_output_tensor(batch_size, height, width);
            
            let id = format!("optimized/{}x{}", width, height);
            group.bench_function(&id, |b| {
                b.iter(|| {
                    conv.forward(&input, &filter, &mut output);
                });
            });
        }
        
        // MPS implementation
        {
            let conv = MetalConv2dMPS::new(
                device.clone(),
                in_channels,
                out_channels,
                kernel_size,
                kernel_size,
                stride,
                padding,
                dilation,
            );
            
            let mut output = conv.create_output_tensor(batch_size, height, width);
            conv.update_weights(&filter);
            
            let id = format!("mps/{}x{}", width, height);
            group.bench_function(&id, |b| {
                b.iter(|| {
                    conv.forward(&input, &mut output);
                });
            });
        }
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_metal_conv);
criterion_main!(benches); 