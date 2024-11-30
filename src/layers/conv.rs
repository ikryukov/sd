use std::sync::Arc;

use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t;
use cudarc::cudnn::{sys, ConvDescriptor, ConvForward, Cudnn};
use cudarc::driver::CudaSlice;
use tracing::info;

use crate::tensor::FilterTensor;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_width: usize,
    kernel_height: usize,
    stride: i32,
    padding: i32,
    dilation: i32,

    desc: ConvDescriptor<f32>,
}

impl Conv2d {
    pub fn new(
        cudnn_handle: Arc<Cudnn>,
        in_channels: usize,
        out_channels: usize,
        kernel_width: usize,
        kernel_height: usize,
        stride: i32,
        padding: i32,
        dilation: i32,
    ) -> Self {
        let mut desc = cudarc::cudnn::Cudnn::create_conv2d::<f32>(
            &cudnn_handle,
            [padding, padding],
            [stride, stride],
            [dilation, dilation],
            cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION
        )
        .unwrap();
        desc.set_math_type(sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)
            .unwrap();

        // Initialize `Conv2d` without the `ConvForward`.
        Conv2d {
            in_channels,
            out_channels,
            kernel_height,
            kernel_width,
            stride,
            padding,
            dilation,
            desc,
        }
    }

    pub fn get_workspace_size(&self, input: &Tensor, filter: &FilterTensor, output: &Tensor) -> usize {
        let conv = ConvForward {
            conv: &self.desc,
            x: &input.desc,
            w: &filter.desc,
            y: &output.desc,
        };
        let res = conv
            .get_workspace_size(cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
            .unwrap();
        res
    }

    pub fn forward(
        &self,
        workspace: Option<&mut CudaSlice<u8>>,
        input: &Tensor, filter: &FilterTensor, output: &mut Tensor
    ) {
        info!("conv2d");
        let conv = ConvForward {
            conv: &self.desc,
            x: &input.desc,
            w: &filter.desc,
            y: &output.desc,
        };

        unsafe {
            conv.launch(
                cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                workspace,
                (1.0f32, 0.0f32),
                &input.data,
                &filter.data,
                &mut output.data,
            )
            .unwrap();
        }
    }
}
