# sd
Stable Diffusion inference

## Platform Support

This project supports both CUDA (for Linux/Windows) and Metal (for macOS) backends:

- **CUDA**: Used on Linux and Windows platforms for GPU acceleration
- **Metal**: Used on macOS platforms for GPU acceleration

The codebase automatically selects the appropriate backend based on the platform.

## Features

- Platform-agnostic layer interfaces
- CUDA implementation for Linux/Windows
- Metal implementation for macOS
- Optimized Metal convolution using threadgroup memory
- Metal Performance Shaders (MPS) implementation for maximum performance on macOS

## Usage

### Running the Model

```bash
# Run the model with specified model and weights files
cargo run -- run model.json weights.safetensors
```

### Running Benchmarks

```bash
# Run benchmarks comparing different implementations
cargo run -- benchmark
```

### Platform-Specific Examples

On macOS:
```bash
# Run Metal convolution example
cargo run -- metal-conv
```

On Linux/Windows:
```bash
# Run CUDA convolution example
cargo run -- cuda-conv
```

## Development

### Adding New Layers

When adding new layers, implement the platform-agnostic traits in `src/layers/layer.rs` and provide platform-specific implementations.

### Metal Shaders

Metal shaders are stored in the `metal-shaders` directory and loaded at runtime.

### CUDA Kernels

CUDA kernels are stored in the `cuda-kernels` directory.
