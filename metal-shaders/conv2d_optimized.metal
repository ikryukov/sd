#include <metal_stdlib>
using namespace metal;

// Optimized convolution kernel using threadgroup memory for better cache utilization
kernel void conv2d_optimized(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant int4 &input_shape [[buffer(3)]],  // N, C, H, W
    constant int4 &filter_shape [[buffer(4)]], // O, I, KH, KW
    constant int4 &output_shape [[buffer(5)]], // N, O, OH, OW
    constant int &stride [[buffer(6)]],
    constant int &padding [[buffer(7)]],
    constant int &dilation [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tgsize [[threads_per_threadgroup]])
{
    // Constants for threadgroup memory size
    const int TILE_WIDTH = 16;
    const int TILE_HEIGHT = 16;
    const int MAX_KERNEL_SIZE = 7;
    
    // Shared memory for input tile with padding for kernel
    threadgroup float input_tile[TILE_HEIGHT + MAX_KERNEL_SIZE - 1][TILE_WIDTH + MAX_KERNEL_SIZE - 1];
    
    // Get output coordinates
    int n = gid.z / output_shape.y;
    int o = gid.z % output_shape.y;
    int oh = gid.y;
    int ow = gid.x;
    
    // Check bounds
    if (n >= output_shape.x || o >= output_shape.y || 
        oh >= output_shape.z || ow >= output_shape.w) {
        return;
    }
    
    // Calculate output index
    int output_idx = ((n * output_shape.y + o) * output_shape.z + oh) * output_shape.w + ow;
    
    // Initialize accumulator
    float acc = 0.0f;
    
    // Kernel dimensions
    int kh = filter_shape.z;
    int kw = filter_shape.w;
    
    // Ensure kernel size is within limits
    if (kh > MAX_KERNEL_SIZE || kw > MAX_KERNEL_SIZE) {
        // Fall back to non-tiled version for large kernels
        for (int i = 0; i < filter_shape.y; i++) {
            for (int ky = 0; ky < kh; ky++) {
                for (int kx = 0; kx < kw; kx++) {
                    // Calculate input position with padding
                    int ih = oh * stride - padding + ky * dilation;
                    int iw = ow * stride - padding + kx * dilation;
                    
                    // Skip if outside input bounds
                    if (ih < 0 || ih >= input_shape.z || iw < 0 || iw >= input_shape.w) {
                        continue;
                    }
                    
                    // Get input value
                    int input_idx = ((n * input_shape.y + i) * input_shape.z + ih) * input_shape.w + iw;
                    float input_val = input[input_idx];
                    
                    // Get weight value
                    int weight_idx = ((o * filter_shape.y + i) * filter_shape.z + ky) * filter_shape.w + kx;
                    float weight_val = weights[weight_idx];
                    
                    // Accumulate
                    acc += input_val * weight_val;
                }
            }
        }
    } else {
        // Use tiled approach for small kernels
        // Process each input channel
        for (int i = 0; i < filter_shape.y; i++) {
            // Calculate base input position for this tile
            int tile_start_h = tgid.y * TILE_HEIGHT;
            int tile_start_w = tgid.x * TILE_WIDTH;
            
            // Load input tile into threadgroup memory (collaborative loading)
            for (int y = tid.y; y < TILE_HEIGHT + kh - 1; y += tgsize.y) {
                for (int x = tid.x; x < TILE_WIDTH + kw - 1; x += tgsize.x) {
                    int ih = tile_start_h + y - padding;
                    int iw = tile_start_w + x - padding;
                    
                    // Check bounds and load from global memory
                    if (ih >= 0 && ih < input_shape.z && iw >= 0 && iw < input_shape.w) {
                        int input_idx = ((n * input_shape.y + i) * input_shape.z + ih) * input_shape.w + iw;
                        input_tile[y][x] = input[input_idx];
                    } else {
                        input_tile[y][x] = 0.0f; // Zero padding
                    }
                }
            }
            
            // Ensure all threads have loaded their part of the tile
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute convolution for this thread's output pixel using the tile
            int local_oh = oh - tile_start_h;
            int local_ow = ow - tile_start_w;
            
            // Only compute if this thread is within the current tile
            if (local_oh >= 0 && local_oh < TILE_HEIGHT && local_ow >= 0 && local_ow < TILE_WIDTH) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        int y = local_oh * stride + ky * dilation;
                        int x = local_ow * stride + kx * dilation;
                        
                        // Get weight value
                        int weight_idx = ((o * filter_shape.y + i) * filter_shape.z + ky) * filter_shape.w + kx;
                        float weight_val = weights[weight_idx];
                        
                        // Accumulate from threadgroup memory
                        acc += input_tile[y][x] * weight_val;
                    }
                }
            }
            
            // Ensure all threads are done with the tile before loading the next one
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // Write output
    output[output_idx] = acc;
} 