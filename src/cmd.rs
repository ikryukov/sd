use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Commands>,
    
    #[arg(long, help = "Path to the model file")]
    pub model: Option<PathBuf>,
    
    #[arg(long, help = "Path to the weights file")]
    pub weights: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run the standard model
    Run {
        #[arg(help = "Path to the model file")]
        model: PathBuf,
        
        #[arg(help = "Path to the weights file")]
        weights: PathBuf,
    },
    
    /// Run the Metal convolution example (macOS only)
    #[cfg(target_os = "macos")]
    MetalConv,
    
    /// Run the CUDA convolution example (non-macOS only)
    #[cfg(not(target_os = "macos"))]
    CudaConv,
    
    /// Run platform-agnostic benchmark comparing available implementations
    Benchmark {
        #[arg(long, help = "Number of iterations to run")]
        iterations: Option<usize>,
        
        #[arg(long, help = "Input size to benchmark (default: 224)")]
        input_size: Option<usize>,
    },
}
