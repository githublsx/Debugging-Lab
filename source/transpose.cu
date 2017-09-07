#include <stdio.h>
#include <iostream>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//NVTX Dir: C:\Program Files\NVIDIA GPU Computing Toolkit\nvToolsExt
#include <nvToolsExt.h>

//Initialize sizes
const int sizeX = 4096;
const int sizeY = 4096;

using namespace std;

struct DIMS
{
    dim3 dimBlock;
    dim3 dimGrid;
};

#define CUDA(call) do {                             \
    cudaError_t e = (call);                         \
    if (e == cudaSuccess) break;                    \
    fprintf(stderr, __FILE__":%d: %s (%d)\n",       \
            __LINE__, cudaGetErrorString(e), e);    \
    exit(1);                                        \
} while (0)

inline unsigned divup(unsigned n, unsigned div)
{
    return (n + div - 1) / div;
}

// Check errors
void postprocess(const float *ref, const float *res, int n)
{
    bool passed = true;
    for (int i = 0; i < n; i++)
    {
        if (res[i] != ref[i])
        {
            printf("ID:%d \t Res:%f \t Ref:%f\n", i, res[i], ref[i]);
            printf("%25s\n", "*** FAILED ***");
            passed = false;
            break;
        }
    }
    if(passed)
        printf("Post process check passed!!\n");
}

void preprocess(float *res, float *dev_res, int n)
{
    std::fill(res, res + n, -1);
    cudaMemset(dev_res, -1, n * sizeof(float));
}

__global__ void copyKernel(const float* const a, float* const b)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;          // Compute correctly - Global X index
    int j = threadIdx.y + blockDim.y * blockIdx.y;          // Compute correctly - Global Y index

    int index = i + sizeX * j;      // Compute 1D index from i, j

    b[index] = a[index];
}

__global__ void matrixTransposeNaive(const float* const a, float* const b)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;          // Compute correctly - Global X index
    int j = threadIdx.y + blockDim.y * blockIdx.y;          // Compute correctly - Global Y index

    int index_in  = i + sizeX * j;  // Compute input index (i,j) from matrix A
    int index_out = i * sizeY + j;  // Compute output index (j,i) in matrix B = transpose(A)

    // Copy data from A to B
    b[index_out] = a[index_in];
}

template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void matrixTransposeShared(const float* const a, float* const b)
{
    //Allocate appropriate shared memory
    __shared__ float mat[BLOCK_SIZE_Y][BLOCK_SIZE_X];

    //Compute input and output index
    int bx = blockIdx.x * blockDim.x;     // Compute block offset - this is number of global threads in X before this block
    int by = blockIdx.y * blockDim.y;     // Compute block offset - this is number of global threads in Y before this block
    int i  = threadIdx.x + bx;     // Global input x index - Same as previous kernels
    int j  = threadIdx.y + by;     // Global input y index - Same as previous kernels
    int ti = threadIdx.x + by;     // Global output x index - remember the transpose
    int tj = threadIdx.y + bx;     // Global output y index - remember the transpose

    //Copy data from input to shared memory
	mat[threadIdx.y][threadIdx.x] = a[i + sizeX * j];

	__syncthreads();

    //Copy data from shared memory to global memory b
	b[tj * sizeX + ti] = mat[threadIdx.x][threadIdx.y];
}

__global__ void matrixTransposeDynamicShared(const float* const a, float* const b)
{
    //Allocate appropriate shared memory
    extern __shared__ float mat[];

    //Compute input and output index - same as matrixTransposeShared kernel
	int bx = blockIdx.x * blockDim.x;     // Compute block offset - this is number of global threads in X before this block
	int by = blockIdx.y * blockDim.y;     // Compute block offset - this is number of global threads in Y before this block
	int i = threadIdx.x + bx;     // Global input x index - Same as previous kernels
	int j = threadIdx.y + by;     // Global input y index - Same as previous kernels
	int ti = threadIdx.x + by;     // Global output x index - remember the transpose
	int tj = threadIdx.y + bx;     // Global output y index - remember the transpose

	//Copy data from input to shared memory
	mat[threadIdx.y * blockDim.x + threadIdx.x] = a[i + sizeX * j];

	__syncthreads();

	//Copy data from shared memory to global memory b
	b[tj * sizeX + ti] = mat[threadIdx.x * blockDim.x + threadIdx.y];
}

int main(int argc, char *argv[])
{
    // Host arrays.
    float* a      = new float[sizeX * sizeY];
    float* b      = new float[sizeX * sizeY];
    float* a_gold = new float[sizeX * sizeY];
    float* b_gold = new float[sizeX * sizeY];

    // Device arrays
    float *d_a, *d_b;

    // Allocate memory on the device
    CUDA(cudaMalloc((void **) &d_a, sizeX * sizeY * sizeof(float)));

    CUDA(cudaMalloc((void **) &d_b, sizeX * sizeY * sizeof(float)));

    // Fill matrix A
    for (int i = 0; i < sizeX * sizeY; i++)
        a[i] = (float)i;

    cout << endl;

    // Copy array contents of A from the host (CPU) to the device (GPU)
    cudaMemcpy(d_a, a, sizeX * sizeY * sizeof(float), cudaMemcpyHostToDevice);

    //Compute "gold" reference standard
    for(int jj = 0; jj < sizeY; jj++)
    {
        for(int ii = 0; ii < sizeX; ii++)
        {
            a_gold[jj * sizeX + ii] = a[jj * sizeX + ii];
            b_gold[ii * sizeY + jj] = a[jj * sizeX + ii];
        }
    }

    cudaDeviceSynchronize();

#define CPU_TRANSPOSE
#ifdef CPU_TRANSPOSE
    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***CPU Transpose***" << endl;
    {
        for (int jj = 0; jj < sizeY; jj++)
            for (int ii = 0; ii < sizeX; ii++)
                b[ii * sizeX + jj] = a[jj * sizeX + ii];
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////
#endif

    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Device To Device Copy***" << endl;
    {
        preprocess(b, d_b, sizeX * sizeY);
        // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"

        DIMS dims;
        dims.dimBlock = dim3(16, 16, 1);
        dims.dimGrid  = dim3(sizeX / dims.dimBlock.x,
                             sizeY / dims.dimBlock.y,
                             1);

        // Launch the GPU kernel
        copyKernel<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, sizeX * sizeY * sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(a_gold, b, sizeX * sizeY);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Naive Transpose***" << endl;
    {
        preprocess(b, d_b, sizeX * sizeY);
        // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"
        // HINT: Look above for copy kernel dims computation
        DIMS dims;
        dims.dimBlock = dim3(16, 16, 1);
        dims.dimGrid  = dim3(sizeX / dims.dimBlock.x,
                             sizeY / dims.dimBlock.y,
                             1);

        // Launch the GPU kernel
        matrixTransposeNaive<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, sizeX * sizeY * sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(b_gold, b, sizeX * sizeY);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Shared Memory Transpose***" << endl;
    {
        preprocess(b, d_b, sizeX * sizeY);
        // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"
        DIMS dims;
        dims.dimBlock = dim3(16, 16, 1);
        dims.dimGrid  = dim3(sizeX / dims.dimBlock.x,
                             sizeY / dims.dimBlock.y,
                             1);

        // Launch the GPU kernel
        matrixTransposeShared<16, 16><<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, sizeX * sizeY * sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(b_gold, b, sizeX * sizeY);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Shared Memory Transpose with Dynamic Shared Memory***" << endl;
    {
        preprocess(b, d_b, sizeX * sizeY);
        // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"
        DIMS dims;
        dims.dimBlock = dim3(16, 16, 1);
        dims.dimGrid  = dim3(sizeX / dims.dimBlock.x,
                             sizeY / dims.dimBlock.y,
                             1);

        // Launch the GPU kernel
        int sharedMemoryPerBlockInBytes = dims.dimBlock.x * dims.dimBlock.y * sizeof(float); // Compute This
        matrixTransposeDynamicShared<<<dims.dimGrid, dims.dimBlock, sharedMemoryPerBlockInBytes>>>(d_a, d_b);
		// Call kernel - 

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, sizeX * sizeY * sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(b_gold, b, sizeX * sizeY);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    // free host memory
    delete[] a;
    delete[] b;

    //CUDA Reset for NVProf
    CUDA(cudaDeviceReset());

    // successful program termination
    return 0;
}
