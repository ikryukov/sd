use cudarc::cudnn::TensorDescriptor;
use cudarc::cudnn::FilterDescriptor;
use cudarc::driver::CudaSlice;

#[derive(Debug)]
pub struct Tensor {
    // NCHW
    pub desc: TensorDescriptor<f32>,
    pub data: CudaSlice<f32>,
}

#[derive(Debug)]
pub struct FilterTensor {
    pub desc: FilterDescriptor<f32>,
    pub data: CudaSlice<f32>,
}