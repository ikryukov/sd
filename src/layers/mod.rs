// Common modules
pub mod layer;

// Platform-specific modules
#[cfg(not(target_os = "macos"))]
pub mod conv;

#[cfg(not(target_os = "macos"))]
pub mod norm;

#[cfg(not(target_os = "macos"))]
pub mod silu;

// Metal-specific modules (macOS only)
#[cfg(target_os = "macos")]
pub mod metal_conv;

#[cfg(target_os = "macos")]
pub mod metal_utils;

#[cfg(target_os = "macos")]
pub mod metal_conv_mps;

// Re-exports for platform-agnostic usage
#[cfg(target_os = "macos")]
pub use metal_conv as conv;

// Ensure norm and silu are available on macOS too
#[cfg(target_os = "macos")]
pub mod norm {
    use crate::layers::layer::{Layer, NormLayer};
    use tracing::info;
    
    #[derive(Debug)]
    pub struct BatchNorm2d {
        num_features: usize,
        epsilon: f32,
    }
    
    impl BatchNorm2d {
        pub fn new(num_features: usize, epsilon: f32) -> Self {
            Self {
                num_features,
                epsilon,
            }
        }
    }
    
    impl Layer for BatchNorm2d {
        fn forward(&self) {
            info!("Metal BatchNorm2d forward pass (placeholder)");
        }
    }
    
    impl NormLayer for BatchNorm2d {
        fn num_features(&self) -> usize {
            self.num_features
        }
        
        fn epsilon(&self) -> f32 {
            self.epsilon
        }
    }
}

#[cfg(target_os = "macos")]
pub mod silu {
    use crate::layers::layer::{Layer, ActivationLayer};
    use tracing::info;
    
    #[derive(Debug)]
    pub struct SiLU;
    
    impl SiLU {
        pub fn new() -> Self {
            Self
        }
    }
    
    impl Layer for SiLU {
        fn forward(&self) {
            info!("Metal SiLU forward pass (placeholder)");
        }
    }
    
    impl ActivationLayer for SiLU {
        fn activation_type(&self) -> &str {
            "silu"
        }
    }
}