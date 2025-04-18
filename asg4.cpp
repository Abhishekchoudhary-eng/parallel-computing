//Q1(a)
#include <cstdio>
#include <cuda_runtime.h>

__global__ void iterativeSum(int* input, int* output, int n) {
    extern __shared__ int s_data[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    s_data[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2*stride) == 0) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = s_data[0];
}

int main() {
    const int N = 1024;
    int *h_input = new int[N], h_output = 0;
    int *d_input, *d_output;

    // Initialize array
    for (int i=0; i<N; ++i) h_input[i] = i+1;

    cudaMalloc(&d_input, N*sizeof(int));
    cudaMalloc(&d_output, sizeof(int));
    cudaMemcpy(d_input, h_input, N*sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    iterativeSum<<<blocks, threads, threads*sizeof(int)>>>(d_input, d_output, N);
    
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum: %d\n", h_output);

    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input;
    return 0;
}

//(b)
__global__ void formulaSum(int* output, int n) {
    *output = n * (n + 1) / 2;
}

// Usage in main():
formulaSum<<<1,1>>>(d_output, N);


//Q2(a)
#include <algorithm>
#include <omp.h>

void parallelMergeSort(float* arr, int n) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int width=1; width<n; width*=2) {
                #pragma omp taskloop
                for (int i=0; i<n; i+=2*width) {
                    int left = i;
                    int mid = i + width;
                    int right = std::min(i+2*width, n);
                    std::inplace_merge(arr+left, arr+mid, arr+right);
                }
            }
        }
    }
}


//(b)
__device__ void merge(int* arr, int l, int m, int r) {
    int temp[r-l+1];
    int i = l, j = m+1, k = 0;
    while (i <= m && j <= r)
        temp[k++]

