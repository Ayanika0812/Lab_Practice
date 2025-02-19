/*
Write a program in CUDA to perform odd even transposition sort in parallel.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10  // Size of the array

__global__ void oddEvenSortKernel(int *arr, int n, int phase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i = 2 * idx + phase; // Determines whether it's an odd or even phase
    
    if (i < n - 1) {
        if (arr[i] > arr[i + 1]) {
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
}

void oddEvenSort(int *h_arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);
    cudaMalloc((void **)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;

    for (int phase = 0; phase < n; phase++) {
        oddEvenSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n, phase % 2);
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
    
    oddEvenSort(h_arr, N);
    
    printf("Sorted array: ");
    for (int i = 0; i < N; i++) printf("%d ", h_arr[i]);
    printf("\n");
    
    return 0;
}
