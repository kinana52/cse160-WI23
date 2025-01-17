#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <stdio.h>

#define BLOCK_WIDTH 8

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  int w = blockIdx.x*blockDim.x+threadIdx.x;
  int h = blockIdx.y*blockDim.y+threadIdx.y;
  int b = blockIdx.z*blockDim.z+threadIdx.z;

  if(b < B){ //for every image in the batch
    for(int m = 0; m < M; m++){ //for every output feature map

      if(h < H_out){ 
        if(w < W_out){//height x width = every element in output
          y4d(b, m, h, w) = 0;
          for(int c = 0; c < C; c++){ //sum over all input feature maps
            for(int p = 0; p < K; p++){ 
              for(int q = 0; q < K; q++){//KxK filter
                y4d(b, m, h, w) += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
              }
            }
          }
        }
      }

    }
  }
 

#undef y4d
#undef x4d
#undef k4d
}


	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    //allocating memory in GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    cudaMalloc( (void**)device_y_ptr, sizeof(float)*H_out*W_out*M*B);
    cudaMalloc( (void**)device_x_ptr, sizeof(float)*H*W*C*B); 
    cudaMalloc( (void**)device_k_ptr, sizeof(float)*K*K*C*M);

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    //copy data to GPU ************WHY DOES THIS WORK FOR DOUBLE POINTER, USUALLY ITS SINGLE POINTER***************
    cudaMemcpy(*device_y_ptr, host_y, sizeof(float)*H_out*W_out*M*B, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_x_ptr, host_x, sizeof(float)*H*W*C*B, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, sizeof(float)*K*K*C*M, cudaMemcpyHostToDevice);

    std::cout << ("in prolog") << std::endl;
    // Useful snippet for error checking
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"Second one:"<<std::endl;
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    std::cout << ("top of kerenel call") << std::endl;
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);                   //I think the kernel takes care of the third dimension? for block and grid
    dim3 gridDim( ceil(W_out/(1.0*BLOCK_WIDTH)), ceil(H_out/(1.0*BLOCK_WIDTH)), ceil(B/(1.0*BLOCK_WIDTH)));    //i think batch size grids needed?
    std::cout<<("here in the kernel")<<std::endl;
    conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    cudaDeviceSynchronize();
    std::cout << "out of kernel" << std::endl;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Copy the output back to host
    cudaMemcpy(host_y, device_y, sizeof(float)*M*H_out*W_out*B, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
