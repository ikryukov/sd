use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use tracing::{error, info};

use crate::tensor::Tensor;

use cuda_kernels::SILU;

#[derive(Debug)]
pub struct Silu {
    device: Arc<CudaDevice>,
}

impl Silu {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Silu { device }
    }

    pub fn forward(&self, input: &Tensor, size: usize, output: &mut Tensor) {
        info!("silu");

        let _ = self
            .device
            .load_ptx(SILU.into(), "silu_forward", &["silu_forward"])
            .map_err(|cuda| error!("{:?}", cuda));
        let f = self
            .device
            .get_func("silu_forward", "silu_forward")
            .unwrap();

        let cfg = LaunchConfig::for_num_elems(size as u32);
        unsafe {
            let _ = f.launch(cfg, (&input.data, &mut output.data, size));
        }
    }
}

#[test]
fn test_silu() {
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

    let silu = Silu::new(device.clone());

    silu.forward(&input_tensor, channels * height * width, &mut output_tensor);

    let host_out = device.dtoh_sync_copy(&output_tensor.data).unwrap();
    println!("{:?}", host_out);

    println!("Silu test passed!");
}
