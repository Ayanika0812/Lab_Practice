#include <stdio.h>
#include <cuda.h>

// CUDA kernel for Sparse Matrix-Vector multiplication using CSR format
__global__ void spmv_csr(int *row_ptr, int *col_idx, float *values, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

int main() {
    // Example sparse matrix in CSR format
    const int num_rows = 4;
    const int num_cols = 4;
    const int num_nonzeros = 6;
    
    int h_row_ptr[] = {0, 2, 4, 5, 6};
    int h_col_idx[] = {0, 1, 1, 2, 2, 3};
    float h_values[] = {10, 20, 30, 40, 50, 60};
    float h_x[] = {1, 2, 3, 4};
    float h_y[num_rows] = {0};
    
    // Device memory allocation
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_idx, num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_values, num_nonzeros * sizeof(float));
    cudaMalloc((void**)&d_x, num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    spmv_csr<<<gridSize, blockSize>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
    
    // Copy result back to host
    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Result vector y:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f\n", h_y[i]);
    }
    
    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
