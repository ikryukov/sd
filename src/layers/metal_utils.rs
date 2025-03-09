use metal::{Device, MTLResourceOptions};

#[cfg(not(target_os = "macos"))]
use cudarc::driver::CudaSlice;

use crate::layers::metal_conv::{MetalTensor, MetalFilterTensor};
use crate::tensor::{Tensor, FilterTensor};

/// Load a Metal shader from a file
pub fn load_metal_shader(device: &Device, shader_path: &std::path::Path) -> metal::Library {
    tracing::info!("Loading Metal shader from: {}", shader_path.display());
    
    // Read the shader source from file
    let shader_source = match std::fs::read_to_string(shader_path) {
        Ok(source) => source,
        Err(e) => {
            panic!("Failed to read shader file {}: {}", shader_path.display(), e);
        }
    };
    
    // Compile the shader
    match device.new_library_with_source(&shader_source, &metal::CompileOptions::new()) {
        Ok(library) => library,
        Err(e) => {
            panic!("Failed to compile shader {}: {}", shader_path.display(), e);
        }
    }
}

/// Convert a Tensor to a MetalTensor
pub fn tensor_to_metal(device: &Device, tensor: &Tensor) -> MetalTensor {
    // Extract shape information from the tensor
    let shape = tensor.get_shape().to_vec();
    
    // Create a Metal buffer with the same size
    let size = shape.iter().product::<usize>();
    
    // Get data from tensor
    let host_data = match std::env::consts::OS {
        "macos" => {
            // On macOS, the tensor already has Metal data
            let buffer_ptr = tensor.data.contents() as *const f32;
            let mut data = vec![0.0f32; size];
            unsafe {
                std::ptr::copy_nonoverlapping(buffer_ptr, data.as_mut_ptr(), size);
            }
            data
        },
        _ => panic!("tensor_to_metal should only be called on macOS")
    };
    
    // Create new Metal buffer
    let buffer = device.new_buffer_with_data(
        host_data.as_ptr() as *const _,
        (size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    
    MetalTensor {
        data: buffer,
        shape,
    }
}

/// Convert a MetalTensor to a Tensor
pub fn metal_to_tensor(metal_tensor: &MetalTensor) -> Tensor {
    // Extract shape information
    let shape = metal_tensor.shape.clone();
    
    // Copy data from Metal to CPU
    let size = shape.iter().product::<usize>();
    let mut host_data = vec![0.0f32; size];
    let buffer_contents = metal_tensor.data.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(buffer_contents, host_data.as_mut_ptr(), size);
    }
    
    // Create tensor with Metal buffer
    Tensor {
        data: metal_tensor.data.clone(),
        shape,
    }
}

/// Convert a FilterTensor to a MetalFilterTensor
pub fn filter_to_metal(device: &Device, filter: &FilterTensor) -> MetalFilterTensor {
    // Extract shape information from the filter
    let shape = filter.get_shape().to_vec();
    
    // Create a Metal buffer with the same size
    let size = shape.iter().product::<usize>();
    
    // Get data from filter
    let host_data = match std::env::consts::OS {
        "macos" => {
            // On macOS, the filter already has Metal data
            let buffer_ptr = filter.data.contents() as *const f32;
            let mut data = vec![0.0f32; size];
            unsafe {
                std::ptr::copy_nonoverlapping(buffer_ptr, data.as_mut_ptr(), size);
            }
            data
        },
        _ => panic!("filter_to_metal should only be called on macOS")
    };
    
    // Create new Metal buffer
    let buffer = device.new_buffer_with_data(
        host_data.as_ptr() as *const _,
        (size * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    
    MetalFilterTensor {
        data: buffer,
        shape,
    }
}

/// Convert a MetalFilterTensor to a FilterTensor
pub fn metal_to_filter(metal_filter: &MetalFilterTensor) -> FilterTensor {
    // Extract shape information
    let shape = metal_filter.shape.clone();
    
    // Copy data from Metal to CPU
    let size = shape.iter().product::<usize>();
    let mut host_data = vec![0.0f32; size];
    let buffer_contents = metal_filter.data.contents() as *const f32;
    unsafe {
        std::ptr::copy_nonoverlapping(buffer_contents, host_data.as_mut_ptr(), size);
    }
    
    // Create filter tensor with Metal buffer
    FilterTensor {
        data: metal_filter.data.clone(),
        shape,
    }
} 