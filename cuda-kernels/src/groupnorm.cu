extern "C" __global__ void group_norm_forward(
    const float *input, float *output,
    const float *gamma, const float *beta,
    int channels, int group_size, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < channels * group_size)
    {
        int group_id = idx / group_size;
        int local_idx = idx % group_size;

        float mean = 0.0f;
        float var = 0.0f;

        for (int i = 0; i < group_size; i++)
        {
            float val = input[group_id * group_size + i];
            mean += val;
        }
        mean /= group_size;

        for (int i = 0; i < group_size; i++)
        {
            float val = input[group_id * group_size + i] - mean;
            var += val * val;
        }
        var = sqrt(var / group_size + epsilon);

        float norm_val = (input[idx] - mean) / var;
        output[idx] = norm_val * gamma[group_id] + beta[group_id];
    }
}
