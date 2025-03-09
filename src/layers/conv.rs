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
    algo: cudnnConvolutionFwdAlgo_t,
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
        // Validate parameters
        assert!(in_channels > 0, "in_channels must be positive");
        assert!(out_channels > 0, "out_channels must be positive");
        assert!(kernel_width > 0, "kernel_width must be positive");
        assert!(kernel_height > 0, "kernel_height must be positive");
        assert!(stride > 0, "stride must be positive");
        assert!(padding >= 0, "padding must be non-negative");
        assert!(dilation > 0, "dilation must be positive");

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

        // Default to WINOGRAD algorithm, but this can be changed
        Conv2d {
            in_channels,
            out_channels,
            kernel_height,
            kernel_width,
            stride,
            padding,
            dilation,
            algo: cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            desc,
        }
    }

    pub fn set_algorithm(&mut self, algo: cudnnConvolutionFwdAlgo_t) {
        self.algo = algo;
    }

    pub fn get_workspace_size(&self, input: &Tensor, filter: &FilterTensor, output: &Tensor) -> usize {
        let conv = ConvForward {
            conv: &self.desc,
            x: &input.desc,
            w: &filter.desc,
            y: &output.desc,
        };
        let res = conv
            .get_workspace_size(self.algo)
            .unwrap_or_else(|e| {
                // Fall back to IMPLICIT_GEMM if the requested algorithm fails
                info!("Algorithm failed with error: {:?}, falling back to IMPLICIT_GEMM", e);
                conv.get_workspace_size(cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
                    .unwrap_or(0)
            });
        res
    }

    pub fn forward(
        &self,
        workspace: Option<&mut CudaSlice<u8>>,
        input: &Tensor, filter: &FilterTensor, output: &mut Tensor
    ) {
        info!("conv2d forward with algorithm: {:?}", self.algo);
        let conv = ConvForward {
            conv: &self.desc,
            x: &input.desc,
            w: &filter.desc,
            y: &output.desc,
        };

        // Try with the configured algorithm first
        let result = unsafe {
            conv.launch(
                self.algo,
                workspace,
                (1.0f32, 0.0f32),
                &input.data,
                &filter.data,
                &mut output.data,
            )
        };

        // If the algorithm fails and we have no workspace, try with IMPLICIT_GEMM which requires no workspace
        if result.is_err() && workspace.is_none() {
            info!("Algorithm failed, falling back to IMPLICIT_GEMM");
            unsafe {
                conv.launch(
                    cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    None,
                    (1.0f32, 0.0f32),
                    &input.data,
                    &filter.data,
                    &mut output.data,
                )
                .unwrap();
            }
        } else {
            // Either the original call succeeded or we have a workspace but still failed
            result.unwrap();
        }
    }
}
