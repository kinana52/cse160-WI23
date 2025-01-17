#include <gputk.h>

#define block_width 16

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this lab
  __shared__ float subTileA[block_width][block_width];
  __shared__ float subTileB[block_width][block_width];

  //block/thread in x and y direction
  int blockX = blockIdx.x; int threadx = threadIdx.x;
  int blockY = blockIdx.y; int thready = threadIdx.y;

  //get row and column of element of output C to work on (tile-wise)
  int row = blockY * block_width + thready;
  int col = blockX * block_width + threadx;
  float pValue = 0;

  //Loop over the A and B tiles required to compute the P element
  for(int phase = 0; phase < ceil( numAColumns/(float)block_width) && phase < ceil( numBRows/(float)block_width); ++phase){
    //make sure not out of tile index
    if(row < numARows && (phase*block_width + threadx) < numAColumns) {
      subTileA[thready][threadx] = A[row*numAColumns + phase*block_width + threadx];
    } else {
      subTileA[thready][threadx] = 0;
    }
    if(col < numBColumns && (phase*block_width+thready) < numBRows) {
      subTileB[thready][threadx] = B[(phase*block_width+thready)*numBColumns + col];
    } else {
      subTileB[thready][threadx] = 0;
    }
    //wait for all of subtile to be filled 
    __syncthreads();

    //multiply/add up the partial dot products of each tile
    for(int k = 0; k < block_width; ++k){
      pValue += subTileA[thready][k] * subTileB[k][threadx];
    }
    __syncthreads();
  }
  if(row < numCRows && col < numCColumns){
    C[row*numCColumns + col] = pValue;
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  //we want a pointer in device to deviceA in host, and deviceA is a pointer to mem, so double pointer
  cudaMalloc( (void**) &deviceA, numAColumns*numARows*sizeof(float));
  cudaMalloc( (void**) &deviceB, numBRows*numBColumns*sizeof(float));
  cudaMalloc( (void**) &deviceC, numCRows*numCColumns*sizeof(float));

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, numCRows*numCColumns*sizeof(float), cudaMemcpyHostToDevice);
  
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(block_width, block_width, 1);
  dim3 gridDim( ceil( (1.0 * numCColumns)/block_width), ceil( (1.0 * numCRows)/block_width), 1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC,
                                     numARows, numAColumns,
                                     numBRows, numBColumns,
                                     numCRows, numCColumns);

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
