use std::sync::Arc;

use cudarc::cudnn::{sys, ConvForward, FilterDescriptor, TensorDescriptor};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use tracing::{error, info};

use crate::tensor::Tensor;

use cuda_kernels::GROUPNORM;

#[derive(Debug)]
pub struct GroupNorm {
    device: Arc<CudaDevice>,
    channels: usize,
    groups: usize,
    gamma: CudaSlice<f32>,
    beta: CudaSlice<f32>,
}

impl GroupNorm {
    fn new(device: Arc<CudaDevice>, channels: usize, groups: usize) -> Self {
        assert!(
            channels % groups == 0,
            "Channels must be divisible by groups!"
        );

        let gamma = device.htod_copy(vec![1.0f32; channels]).unwrap();
        let beta = device.alloc_zeros::<f32>(channels).unwrap();

        Self {
            device,
            channels,
            groups,
            gamma,
            beta,
        }
    }

    fn forward(&self, input: &Tensor, output: &mut Tensor) {
        info!("group norm");
        let group_size = self.channels / self.groups;
        let eps: f32 = 1e-7f32;

        let _ = self
            .device
            .load_ptx(
                GROUPNORM.into(),
                "group_norm_forward",
                &["group_norm_forward"],
            )
            .map_err(|cuda| error!("{:?}", cuda));
        let f = self
            .device
            .get_func("group_norm_forward", "group_norm_forward")
            .unwrap();

        let cfg = LaunchConfig::for_num_elems(self.channels as u32);
        unsafe {
            let _ = f.launch(
                cfg,
                (
                    &input.data,
                    &mut output.data,
                    &self.gamma,
                    &self.beta,
                    self.channels,
                    group_size,
                    eps,
                ),
            );
        }
    }
}

#[test]
fn test_group_norm() {
    let device = cudarc::driver::CudaDevice::new(0).unwrap();
    let cudnn_handle = cudarc::cudnn::Cudnn::new(device.clone()).unwrap();

    let channels: usize = 8;
    let groups: usize = 4;
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
            [1, (channels / groups) as i32, height as i32, width as i32],
        )
        .unwrap(),
        data: device
            .alloc_zeros::<f32>(channels * height * width)
            .unwrap(),
    };

    let group_norm = GroupNorm::new(device.clone(), channels, groups);

    group_norm.forward(&input_tensor, &mut output_tensor);

    let host_out = device.dtoh_sync_copy(&output_tensor.data).unwrap();
    println!("{:?}", host_out);

    // TODO: compute on CPU
    println!("GroupNorm test passed!");
}
