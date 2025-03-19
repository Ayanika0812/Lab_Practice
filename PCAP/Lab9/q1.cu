#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 4           // Matrix size (WIDTH x WIDTH)
#define BLOCK_WIDTH 2     // Block size

// CUDA Kernel for Matrix Multiplication
__global__ void MatMulKernel(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int *h_A, *h_B, *h_C;  // Host matrices
    int *d_A, *d_B, *d_C;  // Device matrices

    // Allocate memory on the host
    int size = WIDTH * WIDTH * sizeof(int);
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize matrices
    printf("Enter elements of Matrix A (4x4):\n");
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        scanf("%d", &h_A[i]);
    }

    printf("Enter elements of Matrix B (4x4):\n");
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        scanf("%d", &h_B[i]);
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridDim(WIDTH / BLOCK_WIDTH, WIDTH / BLOCK_WIDTH);

    // Launch kernel
    MatMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, WIDTH);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Display result
    printf("\nResult of Matrix Multiplication:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%6d ", h_C[i * WIDTH + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}



/*
#include <stdio.h>
#include <cuda_runtime.h>

#define M 3  // Rows of A and C
#define N 3  // Columns of A and Rows of B
#define P 3  // Columns of B and C

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMul(int *A, int *B, int *C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (row < m && col < p) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];  // Dot product
        }
        C[row * p + col] = sum;
    }
}

// Function to print matrix
void printMatrix(int *M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", M[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int size_A = M * N * sizeof(int);
    int size_B = N * P * sizeof(int);
    int size_C = M * P * sizeof(int);

    int A[M * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[N * P] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int C[M * P];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Define 2D Grid and 2D Block
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch Kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("Matrix A:\n");
    printMatrix(A, M, N);

    printf("Matrix B:\n");
    printMatrix(B, N, P);

    printf("Resultant Matrix C (A x B):\n");
    printMatrix(C, M, P);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

*/