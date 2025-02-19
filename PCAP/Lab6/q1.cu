/* Write a program in CUDA which performs convolution operation on one dimensional input  
array N of size width using a mask array M of size mask_width to produce the resultant one 
dimensional array P of size width
*/
#include <stdio.h>
#include <cuda_runtime.h>

#define MASK_WIDTH 3

__global__ void convolution1D(int *N, int *M, int *P, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = MASK_WIDTH / 2;
    int sum = 0;

    if (i < width) {
        for (int j = -radius; j <= radius; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < width) {
                sum += N[idx] * M[j + radius];
            }
        }
        P[i] = sum;
    }
}

int main() {
    const int width = 10;
    int h_N[width] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int h_M[MASK_WIDTH] = {1, 0, -1};
    int h_P[width];

    int *d_N, *d_M, *d_P;
    size_t size_N = width * sizeof(int);
    size_t size_M = MASK_WIDTH * sizeof(int);
    
    cudaMalloc((void **)&d_N, size_N);
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_P, size_N);
    
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (width + blockSize - 1) / blockSize;
    convolution1D<<<gridSize, blockSize>>>(d_N, d_M, d_P, width);
    
    cudaMemcpy(h_P, d_P, size_N, cudaMemcpyDeviceToHost);
    
    printf("Resultant array: ");
    for (int i = 0; i < width; i++) {
        printf("%d ", h_P[i]);
    }
    printf("\n");
    
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
    
    return 0;
}
