// Write a program in CUDA to perform selection sort  in parallel.

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10  // Size of the array

__global__ void selectionSortKernel(int *arr, int n, int step) {
    int minIdx = step;
    for (int j = step + 1 + threadIdx.x; j < n; j += blockDim.x) {
        if (arr[j] < arr[minIdx]) {
            minIdx = j;
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        int temp = arr[step];
        arr[step] = arr[minIdx];
        arr[minIdx] = temp;
    }
}

void selectionSort(int *h_arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);
    cudaMalloc((void **)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    for (int step = 0; step < n - 1; step++) {
        selectionSortKernel<<<1, threadsPerBlock>>>(d_arr, n, step);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int h_arr[N] = {10, 3, 2, 7, 5, 8, 4, 6, 1, 9};
    
    printf("Unsorted array: ");
    for (int i = 0; i < N; i++) printf("%d ", h_arr[i]);
    printf("\n");
    
    selectionSort(h_arr, N);
    
    printf("Sorted array: ");
    for (int i = 0; i < N; i++) printf("%d ", h_arr[i]);
    printf("\n");
    
    return 0;
}
