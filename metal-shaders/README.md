# Metal Shaders

This directory contains Metal shader files used by the project.

## Files

- `conv2d.metal`: Implements a basic 2D convolution kernel for the Metal GPU backend.
- `conv2d_optimized.metal`: Implements an optimized 2D convolution kernel using threadgroup memory for better performance.

## Performance Comparison

The project includes benchmarks to compare the performance of different convolution implementations:

1. **Basic Implementation**: A straightforward implementation of 2D convolution.
2. **Optimized Implementation**: Uses threadgroup memory (shared memory) to reduce global memory accesses.
3. **MPS Implementation**: Uses Apple's Metal Performance Shaders (MPS) for maximum performance.

To run the benchmarks:

```
cargo bench
```

To run the example that compares the implementations:

```
cargo run -- metal-conv
```

## Implementation Details

### Basic Implementation

The basic implementation performs a direct convolution where each thread computes one output pixel. It reads input and filter values directly from global memory.

### Optimized Implementation

The optimized implementation uses several techniques to improve performance:

1. **Threadgroup Memory**: Loads input data into fast threadgroup memory to reduce global memory accesses.
2. **Collaborative Loading**: Multiple threads work together to load the input tile.
3. **Tiling**: Processes the input in tiles to maximize cache utilization.
4. **Fallback Mechanism**: Falls back to the basic implementation for large kernel sizes that wouldn't fit in threadgroup memory.

### MPS Implementation

The MPS implementation leverages Apple's highly optimized Metal Performance Shaders library, which includes hardware-specific optimizations for Apple GPUs.

## Usage

These shaders are loaded at runtime by the Metal implementation of the neural network layers.
The shader files must be present in this directory when running the application.

## Development

When modifying these shaders:

1. Make sure to test your changes with the Metal examples (`cargo run -- metal-conv`).
2. Run the benchmarks to ensure your changes improve performance (`cargo bench`).
3. Follow Metal shader best practices for performance optimization.
4. Keep the shader interface consistent with the Rust code that loads and uses them. 