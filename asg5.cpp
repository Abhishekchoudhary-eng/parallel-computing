#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024


__device__ int a[N], b[N], c[N];

__global__ void vectorAddKernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    
    int h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    
    cudaMemcpyToSymbol(a, h_a, N * sizeof(int));
    cudaMemcpyToSymbol(b, h_b, N * sizeof(int));

    
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    cudaEventRecord(start);
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    
    float msTime;
    cudaEventElapsedTime(&msTime, start, stop);
    printf("Kernel execution time: %.3f ms\n", msTime);

    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double theoreticalBW = (2.0 * prop.memoryClockRate * prop.memoryBusWidth) / (8.0 * 1e6);
    printf("Theoretical Bandwidth: %.2f GB/s\n", theoreticalBW);

    
    double bytesTransferred = 3.0 * N * sizeof(int); // 2 reads + 1 write
    double measuredBW = (bytesTransferred / (msTime * 1e-3)) / 1e9;
    printf("Measured Bandwidth: %.2f GB/s\n", measuredBW);

    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    
    cudaMemcpyFromSymbol(h_c, c, N * sizeof(int));
    printf("Verification: %d + %d = %d\n", h_a[N-1], h_b[N-1], h_c[N-1]);

    return 0;
}
