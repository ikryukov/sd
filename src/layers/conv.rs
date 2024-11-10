use tracing::info;

use super::layer::Layer;

#[derive(Debug)]
pub struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_width: usize,
    kernel_height: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_width: usize,
        kernel_height: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Conv2d {
            in_channels,
            out_channels,
            kernel_height,
            kernel_width,
            stride,
            padding,
            dilation,
        }
    }
}
impl Layer for Conv2d {
    fn forward(&self) {
        info!("conv2d");
    }
}
