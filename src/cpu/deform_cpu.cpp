#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor
deform_conv3d_cpu_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const int kernel_t,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_t,
                    const int stride_h,
                    const int stride_w,
                    const int pad_t,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_t,
                    const int dilation_h,
                    const int dilation_w,
                    const int group,
                    const int deformable_group,
                    const int vol2col_step)
{
    AT_ERROR("Not implement on cpu");
}

std::vector<at::Tensor>
deform_conv3d_cpu_backward(const at::Tensor &input,
                     const at::Tensor &weight,
                     const at::Tensor &bias,
                     const at::Tensor &offset,
                     const at::Tensor &grad_output,
                     const int kernel_t,
                     const int kernel_h,
                     const int kernel_w,
                     const int stride_t,
                     const int stride_h,
                     const int stride_w,
                     const int pad_t,
                     const int pad_h,
                     const int pad_w,
                     const int dilation_t,
                     const int dilation_h,
                     const int dilation_w,
                     const int group,
                     const int deformable_group,
                     const int vol2col_step)
{
    AT_ERROR("Not implement on cpu");
}

