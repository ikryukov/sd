use std::fmt::Debug;

/// Platform-agnostic trait for neural network layers
pub trait Layer: Debug {
    /// Forward pass through the layer
    fn forward(&self);
}

/// Platform-agnostic trait for convolution layers
pub trait ConvLayer: Layer {
    /// Get the number of input channels
    fn in_channels(&self) -> usize;
    
    /// Get the number of output channels
    fn out_channels(&self) -> usize;
    
    /// Get the kernel dimensions
    fn kernel_size(&self) -> (usize, usize);
    
    /// Get the stride
    fn stride(&self) -> i32;
    
    /// Get the padding
    fn padding(&self) -> i32;
    
    /// Get the dilation
    fn dilation(&self) -> i32;
}

/// Platform-agnostic trait for normalization layers
pub trait NormLayer: Layer {
    /// Get the number of features
    fn num_features(&self) -> usize;
    
    /// Get the epsilon value
    fn epsilon(&self) -> f32;
}

/// Platform-agnostic trait for activation layers
pub trait ActivationLayer: Layer {
    /// Get the activation type
    fn activation_type(&self) -> &str;
}
