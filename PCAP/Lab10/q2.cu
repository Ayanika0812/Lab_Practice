#include <stdio.h>
#include <cuda.h>

#define N 16  // Input size
#define K 5   // Kernel size
#define BLOCK_SIZE 256

// Constant memory for kernel
__constant__ float d_kernel[K];

// CUDA kernel for 1D convolution using constant memory
__global__ void conv1D(float *d_input, float *d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (idx < n) {
        for (int i = 0; i < K; i++) {
            int input_idx = idx - K / 2 + i;
            if (input_idx >= 0 && input_idx < n) {
                sum += d_input[input_idx] * d_kernel[i];
            }
        }
        d_output[idx] = sum;
    }
}

int main() {
    float h_input[N], h_output[N], h_kernel[K] = {0.2, 0.2, 0.2, 0.2, 0.2};
    float *d_input, *d_output;
    
    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, K * sizeof(float)); // Copy kernel to constant memory
    
    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conv1D<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Output after 1D Convolution:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
