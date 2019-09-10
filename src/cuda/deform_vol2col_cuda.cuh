#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
#include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__device__ scalar_t dmcn_vol2col_bilinear(const scalar_t *bottom_data, const int data_height, const int data_width,
                                      const int time_l, const int height, const int width, scalar_t t, scalar_t h, scalar_t w)
{
  int h_low = floor(h);
  int h_high = h_low + 1;
  int w_low = floor(w);
  int w_high = w_low + 1;
  int t_low = floor(t);
  int t_high = t_low + 1;

  scalar_t lh = h - h_low;
  scalar_t hh = 1 - lh;
  scalar_t lw = w - w_low;
  scalar_t hw = 1 - lw;
  scalar_t lt = t - t_low;
  scalar_t ht = 1 - lt;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0 && t_low >= 0)
    v1 = bottom_data[((t_low * data_height) + h_low) * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1 && t_low >= 0)
    v2 = bottom_data[((t_low * data_height) + h_low) * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0 && t_low >= 0)
    v3 = bottom_data[((t_low * data_height) + h_high) * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && t_low >= 0)
    v4 = bottom_data[((t_low * data_height) + h_high) * data_width + w_high];
  scalar_t v5 = 0;
  if (h_low >= 0 && w_low >= 0 && t_high <= time_l - 1)
    v1 = bottom_data[((t_high * data_height) + h_low) * data_width + w_low];
  scalar_t v6 = 0;
  if (h_low >= 0 && w_high <= width - 1 && t_high <= time_l - 1)
    v2 = bottom_data[((t_high * data_height) + h_low) * data_width + w_high];
  scalar_t v7 = 0;
  if (h_high <= height - 1 && w_low >= 0 && t_high <= time_l - 1)
    v3 = bottom_data[((t_high * data_height) + h_high) * data_width + w_low];
  scalar_t v8 = 0;
  if (h_high <= height - 1 && w_high <= width - 1 && t_high <= time_l - 1)
    v4 = bottom_data[((t_high * data_height) + h_high) * data_width + w_high];

  scalar_t w1 = hh * hw * ht, w2 = hh * lw * ht, w3 = lh * hw * ht, w4 = lh * lw * ht;
  scalar_t w5 = hh * hw * lt, w6 = hh * lw * lt, w7 = lh * hw * lt, w8 = lh * lw * lt;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);
  return val;
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_gradient_weight(scalar_t argmax_t, scalar_t argmax_h, scalar_t argmax_w,
                                          const int t, const int h, const int w, const int time_l, const int height, const int width)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width || argmax_t <= -1 || argmax_t >= time_l)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_t_low = floor(argmax_t);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  int argmax_t_high = argmax_t_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low && t == argmax_t_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w) * (t + 1 - argmax_t);
  if (h == argmax_h_low && w == argmax_w_high && t == argmax_t_low)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w) * (t + 1 - argmax_t);
  if (h == argmax_h_high && w == argmax_w_low && t == argmax_t_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w) * (t + 1 - argmax_t);
  if (h == argmax_h_high && w == argmax_w_high && t == argmax_t_low)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w) * (t + 1 - argmax_t);
  if (h == argmax_h_low && w == argmax_w_low && t == argmax_t_high)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w) * (argmax_t + 1 - t);
  if (h == argmax_h_low && w == argmax_w_high && t == argmax_t_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w) * (argmax_t + 1 - t);
  if (h == argmax_h_high && w == argmax_w_low && t == argmax_t_high)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w) * (argmax_t + 1 - t);
  if (h == argmax_h_high && w == argmax_w_high && t == argmax_t_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w) * (argmax_t + 1 - t);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t dmcn_get_coordinate_weight(scalar_t argmax_t, scalar_t argmax_h, scalar_t argmax_w,
                                            const int time_l, const int height, const int width, const scalar_t *im_data,
                                            const int data_height, const int data_width, const int bp_dir)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_t_low = floor(argmax_t);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  int argmax_t_high = argmax_t_low + 1;

  scalar_t lw = argmax_w - argmax_w_low;
  scalar_t hw = argmax_w_high - argmax_w;
  scalar_t lh = argmax_h - argmax_h_low;
  scalar_t hh = argmax_h_high - argmax_h;
  scalar_t lt = argmax_t - argmax_t_low;
  scalar_t ht = argmax_t_high - argmax_t;

  scalar_t weight = 0;

  // 三线性插值的求导计算
  if (bp_dir == 0) //解读：计算t的offset
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_t_low >= 0) //imdata[t_low, h_low, w_low]
      weight += -1 * hh * hw * im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high < width && argmax_t_low >= 0)//imdata[t_low, h_low, w_high]
      weight += -1 * hh * lw* im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_high];
    if (argmax_h_high < height && argmax_w_low >= 0 && argmax_t_low >= 0)//imdata[t_low, h_high, w_low]
      weight += -1 * lh * hw * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_low];
    if (argmax_h_high < height && argmax_w_high < width && argmax_t_low >= 0)//imdata[t_low, h_high, w_high]
      weight += -1 *lh * lw * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_high];
    
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_t_high < time_l) //imdata[t_high, h_low, w_low]
      weight += hh * hw * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high < width && argmax_t_high < time_l)//imdata[t_high, h_low, w_high]
      weight += hh * lw* im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_high];
    if (argmax_h_high < height && argmax_w_low >= 0 && argmax_t_high < time_l)//imdata[t_high, h_high, w_low]
      weight += lh * hw * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_low];
    if (argmax_h_high < height && argmax_w_high < width && argmax_t_high < time_l)//imdata[t_high, h_high, w_high]
      weight += lh * lw * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_high];
  }
  else if (bp_dir == 1) //解读：计算h的offset
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_t_low >= 0) //imdata[t_low, h_low, w_low]
      weight += -1 * ht * hw * im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high < width && argmax_t_low >= 0)//imdata[t_low, h_low, w_high]
      weight += -1 * ht * lw * im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_high];
    if (argmax_h_high < height && argmax_w_low >= 0 && argmax_t_low >= 0)//imdata[t_low, h_high, w_low]
      weight += ht * hw * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_low];
    if (argmax_h_high < height && argmax_w_high < width && argmax_t_low >= 0)//imdata[t_low, h_high, w_high]
      weight += ht * lw * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_high];
    
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_t_high < time_l) //imdata[t_high, h_low, w_low]
      weight += -1 * lt * hw * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high < width && argmax_t_high < time_l)//imdata[t_high, h_low, w_high]
      weight += -1 * lt * lw * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_high];
    if (argmax_h_high < height && argmax_w_low >= 0 && argmax_t_high < time_l)//imdata[t_high, h_high, w_low]
      weight += lt * hw * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_low];
    if (argmax_h_high < height && argmax_w_high < width && argmax_t_high < time_l)//imdata[t_high, h_high, w_high]
      weight += lt * lw * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_high];
  }
  else if (bp_dir == 2) //解读：计算w的offset
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_t_low >= 0) //imdata[t_low, h_low, w_low]
      weight += -1 * ht * hh * im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high < width && argmax_t_low >= 0)//imdata[t_low, h_low, w_high]
      weight += ht * hh* im_data[(argmax_t_low * data_height + argmax_h_low) * data_width + argmax_w_high];
    if (argmax_h_high < height && argmax_w_low >= 0 && argmax_t_low >= 0)//imdata[t_low, h_high, w_low]
      weight += -1 * ht * lh * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_low];
    if (argmax_h_high < height && argmax_w_high < width && argmax_t_low >= 0)//imdata[t_low, h_high, w_high]
      weight += ht * lh * im_data[(argmax_t_low * data_height + argmax_h_high) * data_width + argmax_w_high];
    
    if (argmax_h_low >= 0 && argmax_w_low >= 0 && argmax_t_high < time_l) //imdata[t_high, h_low, w_low]
      weight += -1 * lt * hh * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high < width && argmax_t_high < time_l)//imdata[t_high, h_low, w_high]
      weight += lt * hh * im_data[(argmax_t_high * data_height + argmax_h_low) * data_width + argmax_w_high];
    if (argmax_h_high < height && argmax_w_low >= 0 && argmax_t_high < time_l)//imdata[t_high, h_high, w_low]
      weight += -1 * lt * lh * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_low];
    if (argmax_h_high < height && argmax_w_high < width && argmax_t_high < time_l)//imdata[t_high, h_high, w_high]
      weight += lt * lh * im_data[(argmax_t_high * data_height + argmax_h_high) * data_width + argmax_w_high];
  }
  

  return weight;
}

template <typename scalar_t>
__global__ void deformable_vol2col_gpu_kernel(const int n,
                                                       const scalar_t *data_im, const scalar_t *data_offset,
                                                       const int time_l, const int height, const int width, 
                                                       const int kernel_t, const int kernel_h, const int kernel_w,
                                                       const int pad_t, const int pad_h, const int pad_w,
                                                       const int stride_t, const int stride_h, const int stride_w,
                                                       const int dilation_t, const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels, const int deformable_group,
                                                       const int time_col, const int height_col, const int width_col,
                                                       scalar_t *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // NOTE(Jiarui XU): different from CharlesShang's implementation, col_buffer is of shape (N, c*kw*kh, oh * ow)
    // here columns is of shape (c*kw*kh, N, oh, ow), need to adapt axis

    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int t_col = (index / width_col / height_col) % time_col;
    const int b_col = (index / width_col / height_col / time_col) % batch_size;
    const int c_im = (index / width_col / height_col / time_col) / batch_size;
    const int c_col = c_im * kernel_t * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    //解读：这里应该是index of input matrix(including padding),就是生成output中index位置所需要进行卷积的输入（3*3的左上方，3维一次类推）
    const int t_in = t_col * stride_t - pad_t;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    scalar_t *data_col_ptr = data_col + (((c_col * batch_size + b_col) * time_l + t_col) * height_col + h_col) * width_col + w_col;
    // const scalar_t* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * time_l * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 3 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;//TODO：这里是2还是3想清楚

    //解读：这里按照核进行遍历，所有数据按顺序放到col中
    for(int k = 0; k < kernel_t; ++k)
    {
      for (int i = 0; i < kernel_h; ++i)
      {
        for (int j = 0; j < kernel_w; ++j)
        {
          //TODO：这个计算不一定对呀。。。我直接扩展的，并不是特别理解，大概是找到offset的值
          const int data_offset_t_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j)) * time_col + t_col) * height_col + h_col) * width_col + w_col;
          const int data_offset_h_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j) + 1) * time_col + t_col) * height_col + h_col) * width_col + w_col;
          const int data_offset_w_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j) + 2) * time_col + t_col) * height_col + h_col) * width_col + w_col;
          const scalar_t offset_t = data_offset_ptr[data_offset_t_ptr];
          const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
          const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
          scalar_t val = static_cast<scalar_t>(0);
          const scalar_t t_im = t_in + k * dilation_t + offset_t;
          const scalar_t h_im = h_in + i * dilation_h + offset_h;
          const scalar_t w_im = w_in + j * dilation_w + offset_w;
          if (h_im > -1 && w_im > -1 && t_im > -1 && h_im < height && w_im < width && t_im < time_l)
          {
            //const scalar_t map_h = i * dilation_h + offset_h;
            //const scalar_t map_w = j * dilation_w + offset_w;
            //const int cur_height = height - h_in;
            //const int cur_width = width - w_in;
            //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
            val = dmcn_vol2col_bilinear(data_im_ptr, height, width, time_l, height, width, t_im, h_im, w_im);//线性插值获得数值
          }
          *data_col_ptr = val;
          data_col_ptr += batch_size * time_col * height_col * width_col;//看函数的注释，应该是因为here columns is of shape (c*kw*kh, N, ot, oh, ow), need to adapt axis
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void deformable_col2vol_gpu_kernel(const int n,
                                                       const scalar_t *data_col, const scalar_t *data_offset,
                                                       const int channels, 
                                                       const int time_l, const int height, const int width, 
                                                       const int kernel_t, const int kernel_h, const int kernel_w,
                                                       const int pad_t, const int pad_h, const int pad_w,
                                                       const int stride_t, const int stride_h, const int stride_w,
                                                       const int dilation_t, const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int deformable_group,
                                                       const int time_col, const int height_col, const int width_col,
                                                       scalar_t *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {//解读：这个函数是用来计算关于输入的feature map的梯度的
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int k = (index / width_col / height_col / batch_size / kernel_w / kernel_h) % kernel_t;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h / kernel_t;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int t_out = (index / width_col / height_col) % time_col;
    int b = (index / width_col / height_col / time_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;
    int t_in = t_out * stride_t - pad_t;

    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 3 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;
    
    const int data_offset_t_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j)) * time_col + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_h_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j) + 1) * time_col + t_out) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j) + 2) * time_col + t_out) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_t = data_offset_ptr[data_offset_t_ptr];
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_t_data = t_in + k * dilation_t + offset_t;
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_t = (int)cur_inv_t_data;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    //TODO:????
    for (int dz = -2; dz <= 2; dz++)
    {
      for (int dy = -2; dy <= 2; dy++)
      {
        for (int dx = -2; dx <= 2; dx++)
        {
          if (cur_t + dz >= 0 && cur_t + dz < time_l &&
              cur_h + dy >= 0 && cur_h + dy < height &&
              cur_w + dx >= 0 && cur_w + dx < width &&
              abs(cur_inv_t_data - (cur_t + dz)) < 1 &&
              abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
              abs(cur_inv_w_data - (cur_w + dx)) < 1)
          {
            int cur_bottom_grad_pos = (((b * channels + c) * time_l + cur_t + dz) * height + cur_h + dy) * width + cur_w + dx;
            scalar_t weight = dmcn_get_gradient_weight(cur_inv_t_data, cur_inv_h_data, cur_inv_w_data, cur_t + dz, cur_h + dy, cur_w + dx,time_l, height, width);
            atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
          }
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void deformable_col2vol_coord_gpu_kernel(const int n,
                                                             const scalar_t *data_col, const scalar_t *data_im,
                                                             const scalar_t *data_offset,
                                                             const int channels, const int time_l, const int height, const int width,
                                                             const int kernel_t, const int kernel_h, const int kernel_w,
                                                             const int pad_t, const int pad_h, const int pad_w,
                                                             const int stride_t, const int stride_h, const int stride_w,
                                                             const int dilation_t, const int dilation_h, const int dilation_w,
                                                             const int channel_per_deformable_group,
                                                             const int batch_size, const int offset_channels, const int deformable_group,
                                                             const int time_col, const int height_col, const int width_col,
                                                             scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {//解读：这个函数是用来计算关于offset的梯度的, 计算offset中第index的梯度
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int t = (index / width_col / height_col) % time_col;
    int c = (index / width_col / height_col / time_col) % offset_channels;
    int b = (index / width_col / height_col / time_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (3 * kernel_t * kernel_h * kernel_w);
    const int col_step = kernel_t * kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * kernel_t * width_col * height_col;
    const scalar_t *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_t / kernel_h / kernel_w * time_l * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 3 * kernel_t * kernel_h * kernel_w * time_col * height_col * width_col;

    const int offset_c = c - deformable_group_index * 3 * kernel_t * kernel_h * kernel_w;

    for (int col_c = (offset_c / 3); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = ((((col_c * batch_size + b) * time_col) + t * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 3;//0：计算t的offset；1：计算h的offet；2：计算w的offset

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int k = (col_pos / width_col / height_col / batch_size / kernel_w / kernel_h) % kernel_t;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int t_out = (col_pos / width_col / height_col) % time_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      int t_in = t_out * stride_t - pad_t;
      //TODO:这里计算要想清楚
      const int data_offset_t_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j)) * time_col + t_out) * height_col + h_out) * width_col + w_out;
      const int data_offset_h_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j) + 1) * time_col + t_out) * height_col + h_out) * width_col + w_out;
      const int data_offset_w_ptr = (((3 * ((k * kernel_h + i) * kernel_w + j) + 2) * time_col + t_out) * height_col + h_out) * width_col + w_out;
      
      const scalar_t offset_t = data_offset_ptr[data_offset_t_ptr];
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_t = t_in + k * dilation_t + offset_t;
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;//边界外？
      }
      const scalar_t weight = dmcn_get_coordinate_weight(
          inv_t, inv_h, inv_w,
          time_l, height, width, data_im_ptr + cnt * time_l * height * width, height , width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    grad_offset[index] = val;
  }
}

template <typename scalar_t>
void deformable_vol2col_cuda(cudaStream_t stream,
  const scalar_t* data_im, const scalar_t* data_offset,
  const int batch_size, const int channels, const int time_im, const int height_im, const int width_im, 
  const int time_col, const int height_col, const int width_col, const int  kernel_t, const int kernel_h, const int kernel_w,
  const int pad_t, const int pad_h, const int pad_w, const int stride_t, const int stride_h, const int stride_w, 
  const int dilation_t, const int dilation_h, const int dilation_w,
  const int deformable_group, scalar_t* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * time_col * height_col * width_col;
  deformable_vol2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, time_im, height_im, width_im, kernel_t, kernel_h, kernel_w,
      pad_t, pad_h, pad_w, stride_t, stride_h, stride_w, dilation_t, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, time_col, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_vol2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void deformable_col2vol_cuda(cudaStream_t stream,
  const scalar_t* data_col, const scalar_t* data_offset,
  const int batch_size, const int channels, const int time_im, const int height_im, const int width_im, 
  const int time_col, const int height_col, const int width_col, const int kernel_t, const int kernel_h, const int kernel_w,
  const int pad_t, const int pad_h, const int pad_w, const int stride_t, const int stride_h, const int stride_w, 
  const int dilation_t, const int dilation_h, const int dilation_w, 
  const int deformable_group, scalar_t* grad_im){

  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_t * kernel_h * kernel_w * batch_size * time_col * height_col * width_col;
  deformable_col2vol_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, data_col, data_offset, channels, time_im, height_im, width_im,
        kernel_t, kernel_h, kernel_w, pad_t, pad_h, pad_h, stride_t, stride_h, stride_w,
        dilation_t, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, deformable_group, time_col, height_col, width_col, grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("fuck:error in deformable_col2vol_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void deformable_col2vol_coord_cuda(cudaStream_t stream,
  const scalar_t* data_col, const scalar_t* data_im, const scalar_t* data_offset,
  const int batch_size, const int channels, const int time_im, const int height_im, const int width_im, 
  const int time_col, const int height_col, const int width_col, const int kernel_t, const int kernel_h, const int kernel_w,
  const int pad_t, const int pad_h, const int pad_w, const int stride_t, const int stride_h, const int stride_w, 
  const int dilation_t, const int dilation_h, const int dilation_w, 
  const int deformable_group,
  scalar_t* grad_offset) {
  const int num_kernels = batch_size * time_col * height_col * width_col * 3 * kernel_t * kernel_h * kernel_w * deformable_group;//TODO:这里是2还是3不确定
  const int channel_per_deformable_group = channels * kernel_t * kernel_h * kernel_w / deformable_group;
  deformable_col2vol_coord_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, data_col, data_im, data_offset, channels, time_im, height_im, width_im,
        kernel_t, kernel_h, kernel_w, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w,
        dilation_t, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, 2 * kernel_t * kernel_h * kernel_w * deformable_group, deformable_group, time_col, height_col, width_col, 
        grad_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("fuck:  error in deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}