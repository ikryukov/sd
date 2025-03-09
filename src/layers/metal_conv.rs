use std::sync::Arc;
use std::path::Path;

use metal::{Device, MTLResourceOptions, MTLSize, ComputePipelineState, Buffer, CommandQueue};
use tracing::info;

use crate::layers::metal_utils::load_metal_shader;
use crate::layers::layer::{Layer, ConvLayer};

/// Enum to select which convolution implementation to use
#[derive(Debug, Clone, Copy)]
pub enum ConvImplementation {
    /// Basic implementation
    Basic,
    /// Optimized implementation with threadgroup memory
    Optimized,
}

// We'll need to create Metal-specific tensor types
#[derive(Debug)]
pub struct MetalTensor {
    pub data: Buffer,
    pub shape: Vec<usize>, // NCHW format
}

#[derive(Debug)]
pub struct MetalFilterTensor {
    pub data: Buffer,
    pub shape: Vec<usize>, // OIHW format (out_channels, in_channels, height, width)
}

#[derive(Debug)]
pub struct MetalConv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_width: usize,
    kernel_height: usize,
    stride: i32,
    padding: i32,
    dilation: i32,
    
    device: Device,
    basic_pipeline: ComputePipelineState,
    optimized_pipeline: Option<ComputePipelineState>,
    command_queue: CommandQueue,
    implementation: ConvImplementation,
}

impl MetalConv2d {
    pub fn new(
        device: Device,
        in_channels: usize,
        out_channels: usize,
        kernel_width: usize,
        kernel_height: usize,
        stride: i32,
        padding: i32,
        dilation: i32,
        implementation: ConvImplementation,
    ) -> Self {
        // Validate parameters
        assert!(in_channels > 0, "in_channels must be positive");
        assert!(out_channels > 0, "out_channels must be positive");
        assert!(kernel_width > 0, "kernel_width must be positive");
        assert!(kernel_height > 0, "kernel_height must be positive");
        assert!(stride > 0, "stride must be positive");
        assert!(padding >= 0, "padding must be non-negative");
        assert!(dilation > 0, "dilation must be positive");

        // Create the Metal compute pipeline
        let command_queue = device.new_command_queue();
        
        // Load the basic Metal shader from file
        let basic_shader_path = Path::new("metal-shaders/conv2d.metal");
        let basic_library = load_metal_shader(&device, basic_shader_path);
        let basic_kernel = basic_library.get_function("conv2d", None).unwrap();
        let basic_pipeline = device.new_compute_pipeline_state_with_function(&basic_kernel).unwrap();
        
        // Load the optimized Metal shader if requested
        let optimized_pipeline = if matches!(implementation, ConvImplementation::Optimized) {
            let optimized_shader_path = Path::new("metal-shaders/conv2d_optimized.metal");
            if optimized_shader_path.exists() {
                let optimized_library = load_metal_shader(&device, optimized_shader_path);
                let optimized_kernel = optimized_library.get_function("conv2d_optimized", None).unwrap();
                Some(device.new_compute_pipeline_state_with_function(&optimized_kernel).unwrap())
            } else {
                info!("Optimized shader not found, falling back to basic implementation");
                None
            }
        } else {
            None
        };
        
        MetalConv2d {
            in_channels,
            out_channels,
            kernel_height,
            kernel_width,
            stride,
            padding,
            dilation,
            device,
            basic_pipeline,
            optimized_pipeline,
            command_queue,
            implementation,
        }
    }
    
    /// Set the implementation to use
    pub fn set_implementation(&mut self, implementation: ConvImplementation) {
        self.implementation = implementation;
        
        // Load the optimized shader if needed and not already loaded
        if matches!(implementation, ConvImplementation::Optimized) && self.optimized_pipeline.is_none() {
            let optimized_shader_path = Path::new("metal-shaders/conv2d_optimized.metal");
            if optimized_shader_path.exists() {
                let optimized_library = load_metal_shader(&self.device, optimized_shader_path);
                let optimized_kernel = optimized_library.get_function("conv2d_optimized", None).unwrap();
                self.optimized_pipeline = Some(self.device.new_compute_pipeline_state_with_function(&optimized_kernel).unwrap());
            } else {
                info!("Optimized shader not found, falling back to basic implementation");
            }
        }
    }
    
    pub fn forward(
        &self,
        input: &MetalTensor,
        filter: &MetalFilterTensor,
        output: &mut MetalTensor,
    ) {
        match self.implementation {
            ConvImplementation::Basic => {
                info!("metal_conv2d forward with basic implementation");
                self.forward_basic(input, filter, output);
            },
            ConvImplementation::Optimized => {
                if let Some(ref optimized_pipeline) = self.optimized_pipeline {
                    info!("metal_conv2d forward with optimized implementation");
                    self.forward_optimized(input, filter, output, optimized_pipeline);
                } else {
                    info!("Optimized pipeline not available, falling back to basic implementation");
                    self.forward_basic(input, filter, output);
                }
            }
        }
    }
    
    fn forward_basic(
        &self,
        input: &MetalTensor,
        filter: &MetalFilterTensor,
        output: &mut MetalTensor,
    ) {
        // Extract shapes
        let batch_size = input.shape[0];
        let input_height = input.shape[2];
        let input_width = input.shape[3];
        
        let output_height = output.shape[2];
        let output_width = output.shape[3];
        
        // Create parameter buffers
        let input_shape: [i32; 4] = [
            batch_size as i32, 
            self.in_channels as i32, 
            input_height as i32, 
            input_width as i32
        ];
        let input_shape_buffer = unsafe {
            self.device.new_buffer_with_data(
                input_shape.as_ptr() as *const std::ffi::c_void,
                (input_shape.len() * std::mem::size_of::<i32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let filter_shape: [i32; 4] = [
            self.out_channels as i32, 
            self.in_channels as i32, 
            self.kernel_height as i32, 
            self.kernel_width as i32
        ];
        let filter_shape_buffer = unsafe {
            self.device.new_buffer_with_data(
                filter_shape.as_ptr() as *const std::ffi::c_void,
                (filter_shape.len() * std::mem::size_of::<i32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let output_shape: [i32; 4] = [
            batch_size as i32, 
            self.out_channels as i32, 
            output_height as i32, 
            output_width as i32
        ];
        let output_shape_buffer = unsafe {
            self.device.new_buffer_with_data(
                output_shape.as_ptr() as *const std::ffi::c_void,
                (output_shape.len() * std::mem::size_of::<i32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let stride_val = self.stride;
        let stride_buffer = unsafe {
            self.device.new_buffer_with_data(
                &stride_val as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let padding_val = self.padding;
        let padding_buffer = unsafe {
            self.device.new_buffer_with_data(
                &padding_val as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let dilation_val = self.dilation;
        let dilation_buffer = unsafe {
            self.device.new_buffer_with_data(
                &dilation_val as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        // Set pipeline and buffers
        compute_encoder.set_compute_pipeline_state(&self.basic_pipeline);
        compute_encoder.set_buffer(0, Some(&input.data), 0);
        compute_encoder.set_buffer(1, Some(&filter.data), 0);
        compute_encoder.set_buffer(2, Some(&output.data), 0);
        compute_encoder.set_buffer(3, Some(&input_shape_buffer), 0);
        compute_encoder.set_buffer(4, Some(&filter_shape_buffer), 0);
        compute_encoder.set_buffer(5, Some(&output_shape_buffer), 0);
        compute_encoder.set_buffer(6, Some(&stride_buffer), 0);
        compute_encoder.set_buffer(7, Some(&padding_buffer), 0);
        compute_encoder.set_buffer(8, Some(&dilation_buffer), 0);
        
        // Calculate grid size
        let threads_per_group = MTLSize::new(8, 8, 1);
        let grid_size = MTLSize::new(
            (output_width as u64 + threads_per_group.width - 1) / threads_per_group.width,
            (output_height as u64 + threads_per_group.height - 1) / threads_per_group.height,
            (batch_size * self.out_channels) as u64,
        );
        
        // Dispatch
        compute_encoder.dispatch_thread_groups(grid_size, threads_per_group);
        compute_encoder.end_encoding();
        
        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    fn forward_optimized(
        &self,
        input: &MetalTensor,
        filter: &MetalFilterTensor,
        output: &mut MetalTensor,
        pipeline: &ComputePipelineState,
    ) {
        // Extract shapes
        let batch_size = input.shape[0];
        let input_height = input.shape[2];
        let input_width = input.shape[3];
        
        let output_height = output.shape[2];
        let output_width = output.shape[3];
        
        // Create parameter buffers
        let input_shape: [i32; 4] = [
            batch_size as i32, 
            self.in_channels as i32, 
            input_height as i32, 
            input_width as i32
        ];
        let input_shape_buffer = unsafe {
            self.device.new_buffer_with_data(
                input_shape.as_ptr() as *const std::ffi::c_void,
                (input_shape.len() * std::mem::size_of::<i32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let filter_shape: [i32; 4] = [
            self.out_channels as i32, 
            self.in_channels as i32, 
            self.kernel_height as i32, 
            self.kernel_width as i32
        ];
        let filter_shape_buffer = unsafe {
            self.device.new_buffer_with_data(
                filter_shape.as_ptr() as *const std::ffi::c_void,
                (filter_shape.len() * std::mem::size_of::<i32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let output_shape: [i32; 4] = [
            batch_size as i32, 
            self.out_channels as i32, 
            output_height as i32, 
            output_width as i32
        ];
        let output_shape_buffer = unsafe {
            self.device.new_buffer_with_data(
                output_shape.as_ptr() as *const std::ffi::c_void,
                (output_shape.len() * std::mem::size_of::<i32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let stride_val = self.stride;
        let stride_buffer = unsafe {
            self.device.new_buffer_with_data(
                &stride_val as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let padding_val = self.padding;
        let padding_buffer = unsafe {
            self.device.new_buffer_with_data(
                &padding_val as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        let dilation_val = self.dilation;
        let dilation_buffer = unsafe {
            self.device.new_buffer_with_data(
                &dilation_val as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        
        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        // Set pipeline and buffers
        compute_encoder.set_compute_pipeline_state(pipeline);
        compute_encoder.set_buffer(0, Some(&input.data), 0);
        compute_encoder.set_buffer(1, Some(&filter.data), 0);
        compute_encoder.set_buffer(2, Some(&output.data), 0);
        compute_encoder.set_buffer(3, Some(&input_shape_buffer), 0);
        compute_encoder.set_buffer(4, Some(&filter_shape_buffer), 0);
        compute_encoder.set_buffer(5, Some(&output_shape_buffer), 0);
        compute_encoder.set_buffer(6, Some(&stride_buffer), 0);
        compute_encoder.set_buffer(7, Some(&padding_buffer), 0);
        compute_encoder.set_buffer(8, Some(&dilation_buffer), 0);
        
        // For the optimized version, use 16x16 thread groups
        let threads_per_group = MTLSize::new(16, 16, 1);
        
        // Calculate grid size based on tile size
        let tile_width = 16;
        let tile_height = 16;
        let grid_size = MTLSize::new(
            (output_width as u64 + tile_width - 1) / tile_width,
            (output_height as u64 + tile_height - 1) / tile_height,
            (batch_size * self.out_channels) as u64,
        );
        
        // Dispatch
        compute_encoder.dispatch_thread_groups(grid_size, threads_per_group);
        compute_encoder.end_encoding();
        
        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    // Helper method to calculate output dimensions
    pub fn calculate_output_size(
        &self,
        input_height: usize,
        input_width: usize,
    ) -> (usize, usize) {
        let output_height = ((input_height as i32 + 2 * self.padding - self.dilation * (self.kernel_height as i32 - 1) - 1) / self.stride) + 1;
        let output_width = ((input_width as i32 + 2 * self.padding - self.dilation * (self.kernel_width as i32 - 1) - 1) / self.stride) + 1;
        
        (output_height as usize, output_width as usize)
    }
    
    // Helper method to create an output tensor with the right dimensions
    pub fn create_output_tensor(
        &self,
        batch_size: usize,
        input_height: usize,
        input_width: usize,
    ) -> MetalTensor {
        let (output_height, output_width) = self.calculate_output_size(input_height, input_width);
        let output_size = batch_size * self.out_channels * output_height * output_width;
        
        let buffer = self.device.new_buffer(
            (output_size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        MetalTensor {
            data: buffer,
            shape: vec![batch_size, self.out_channels, output_height, output_width],
        }
    }
}

impl Layer for MetalConv2d {
    fn forward(&self) {
        // This is a simplified version that doesn't take parameters
        // In a real implementation, we would need to store input, filter, and output tensors as fields
        info!("MetalConv2d forward pass (simplified)");
        
        // In a real implementation, we would call the actual forward method with stored tensors
        // For example:
        // if let (Some(input), Some(filter), Some(output)) = (&self.input, &self.filter, &mut self.output) {
        //     self.forward_impl(input, filter, output);
        // }
    }
}

impl ConvLayer for MetalConv2d {
    fn in_channels(&self) -> usize {
        self.in_channels
    }
    
    fn out_channels(&self) -> usize {
        self.out_channels
    }
    
    fn kernel_size(&self) -> (usize, usize) {
        (self.kernel_height, self.kernel_width)
    }
    
    fn stride(&self) -> i32 {
        self.stride
    }
    
    fn padding(&self) -> i32 {
        self.padding
    }
    
    fn dilation(&self) -> i32 {
        self.dilation
    }
} 