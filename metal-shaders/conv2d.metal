#include <metal_stdlib>
using namespace metal;

kernel void conv2d(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant int4 &input_shape [[buffer(3)]],  // N, C, H, W
    constant int4 &filter_shape [[buffer(4)]], // O, I, KH, KW
    constant int4 &output_shape [[buffer(5)]], // N, O, OH, OW
    constant int &stride [[buffer(6)]],
    constant int &padding [[buffer(7)]],
    constant int &dilation [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
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
    
    // Convolution
    for (int i = 0; i < filter_shape.y; i++) {
        for (int kh = 0; kh < filter_shape.z; kh++) {
            for (int kw = 0; kw < filter_shape.w; kw++) {
                // Calculate input position with padding
                int ih = oh * stride - padding + kh * dilation;
                int iw = ow * stride - padding + kw * dilation;
                
                // Skip if outside input bounds
                if (ih < 0 || ih >= input_shape.z || iw < 0 || iw >= input_shape.w) {
                    continue;
                }
                
                // Get input value
                int input_idx = ((n * input_shape.y + i) * input_shape.z + ih) * input_shape.w + iw;
                float input_val = input[input_idx];
                
                // Get weight value
                int weight_idx = ((o * filter_shape.y + i) * filter_shape.z + kh) * filter_shape.w + kw;
                float weight_val = weights[weight_idx];
                
                // Accumulate
                acc += input_val * weight_val;
            }
        }
    }
    
    // Write output
    output[output_idx] = acc;
} 