#[cfg(not(target_os = "macos"))]
use cudarc::cudnn::TensorDescriptor;
#[cfg(not(target_os = "macos"))]
use cudarc::cudnn::FilterDescriptor;
#[cfg(not(target_os = "macos"))]
use cudarc::driver::CudaSlice;

#[cfg(target_os = "macos")]
use metal::Buffer;

// CUDA-specific tensor implementation
#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
pub struct Tensor {
    // NCHW
    pub desc: TensorDescriptor<f32>,
    pub data: CudaSlice<f32>,
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug)]
pub struct FilterTensor {
    pub desc: FilterDescriptor<f32>,
    pub data: CudaSlice<f32>,
}

// Metal-specific tensor implementation
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct Tensor {
    // NCHW
    pub data: Buffer,
    pub shape: Vec<usize>,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct FilterTensor {
    pub data: Buffer,
    pub shape: Vec<usize>, // OIHW format (out_channels, in_channels, height, width)
}

// Common tensor methods that work across platforms
impl Tensor {
    #[cfg(target_os = "macos")]
    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }
    
    #[cfg(not(target_os = "macos"))]
    pub fn get_shape(&self) -> Vec<usize> {
        self.desc.get_dimensions().unwrap()
    }
}

impl FilterTensor {
    #[cfg(target_os = "macos")]
    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }
    
    #[cfg(not(target_os = "macos"))]
    pub fn get_shape(&self) -> Vec<usize> {
        self.desc.get_dimensions().unwrap()
    }
}