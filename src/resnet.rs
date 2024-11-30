use std::sync::Arc;

use cudarc::{cudnn::Cudnn, driver::CudaDevice};

use crate::{
    layers::{norm::GroupNorm, silu::Silu},
    tensor::{FilterTensor, Tensor},
    Conv2d,
};

#[derive(Debug)]
pub struct ResnetBlock2D {
    pub device: Arc<CudaDevice>,
    pub cudnn: Arc<Cudnn>,
    pub norm1: GroupNorm,
    pub norm2: GroupNorm,
    pub conv1: Conv2d,
    pub conv1_weights: FilterTensor,
    pub conv2: Conv2d,
    pub conv2_weights: FilterTensor,
    pub activation: Silu,

    pub in_channels: usize,
    pub out_channels: usize,
}

impl ResnetBlock2D {
    fn new(
        device: Arc<CudaDevice>,
        cudnn: Arc<Cudnn>,
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        groups_out: usize,
        conv1_weights: Vec<f32>,
        conv2_weights: Vec<f32>,
    ) -> Self {
        ResnetBlock2D {
            device: device.clone(),
            cudnn: cudnn.clone(),
            in_channels,
            out_channels: in_channels,
            norm1: GroupNorm::new(device.clone(), in_channels, groups),
            norm2: GroupNorm::new(device.clone(), in_channels, groups_out),
            conv1: Conv2d::new(cudnn.clone(), in_channels, out_channels, 3, 3, 1, 1, 1),
            conv1_weights: FilterTensor {
                desc: cudarc::cudnn::Cudnn::create_4d_filter(
                    &cudnn,
                    cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    [1, 1, 3, 3],
                )
                .unwrap(),
                data: device.htod_copy(conv1_weights).unwrap(),
            },
            conv2: Conv2d::new(cudnn.clone(), out_channels, out_channels, 3, 3, 1, 1, 1),
            conv2_weights: FilterTensor {
                desc: cudarc::cudnn::Cudnn::create_4d_filter(
                    &cudnn,
                    cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    [1, 1, 3, 3],
                )
                .unwrap(),
                data: device.htod_copy(conv2_weights).unwrap(),
            },
            activation: Silu::new(device.clone()),
        }
    }

    pub fn forward(&self, input: &Tensor, output: &mut Tensor) {
        self.norm1.forward(input, output);
        let size = 8 * 1 * 1;
        let hidden = Tensor {
            desc: cudarc::cudnn::Cudnn::create_4d_tensor::<f32>(
                &self.cudnn,
                cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [1, 8 as i32, 1 as i32, 1 as i32],
            )
            .unwrap(),
            data: output.data.clone(),
        };

        self.activation.forward(&hidden, size, output);

        let workspace_size = self.conv1.get_workspace_size(input, &self.conv1_weights, output);
        let mut workspace = self.device.alloc_zeros::<u8>(workspace_size).unwrap();

        self.conv1.forward(Some(&mut workspace), input, &self.conv1_weights, output);
    }
}

#[test]
fn test_resnetblock2d() {
    let device = cudarc::driver::CudaDevice::new(0).unwrap();
    let cudnn_handle = cudarc::cudnn::Cudnn::new(device.clone()).unwrap();

    let channels: usize = 8;
    let height: usize = 1;
    let width: usize = 1;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];

    let input_tensor = Tensor {
        desc: cudarc::cudnn::Cudnn::create_4d_tensor::<f32>(
            &cudnn_handle,
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [1, channels as i32, height as i32, width as i32],
        )
        .unwrap(),
        data: device.htod_copy(input_data).unwrap(),
    };

    let mut output_tensor = Tensor {
        desc: cudarc::cudnn::Cudnn::create_4d_tensor::<f32>(
            &cudnn_handle,
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [1, channels as i32, height as i32, width as i32],
        )
        .unwrap(),
        data: device
            .alloc_zeros::<f32>(channels * height * width)
            .unwrap(),
    };

    let resnet_block = ResnetBlock2D::new(device.clone(), cudnn_handle.clone(), 8, 8, 4, 4);

    resnet_block.forward(&input_tensor, &mut output_tensor);

    let host_out = device.dtoh_sync_copy(&output_tensor.data).unwrap();
    println!("ResnetBlock2D result: {:?}", host_out);

    println!("ResnetBlock2D test passed!");
}
