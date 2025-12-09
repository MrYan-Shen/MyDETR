#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    // 添加CUDA实现
    AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    // 其他参数检查...

    // 调用CUDA kernel
    // 实际实现可参考原项目的CUDA实现
}