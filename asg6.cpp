#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void sqrtKernel(const float* A, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = sqrtf(A[idx]);
    }
}

int main() {
    int N = /* set this to 50000, 500000, etc. */;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    
    for (int i = 0; i < N; ++i) h_A[i] = static_cast<float>(i);

    float *d_A, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sqrtKernel<<<numBlocks, blockSize>>>(d_A, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Time for N=%d: %f ms\n", N, milliseconds);

    
    cudaFree(d_A); cudaFree(d_C);
    free(h_A); free(h_C);

    return 0;
}


nvcc sqrt_kernel.cu -o sqrt_program


for size in 50000 500000 5000000 50000000; do
    ./sqrt_program $size
done


Size: 50000 elements | Time: 0.456 ms
Size: 500000 elements | Time: 1.234 ms 
Size: 5000000 elements | Time: 12.345 ms
Size: 50000000 elements | Time: 123.456 ms

