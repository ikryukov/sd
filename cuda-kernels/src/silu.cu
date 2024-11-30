extern "C" __global__ void silu_forward(const float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x)); // SiLU: x * sigmoid(x)
    }
}
