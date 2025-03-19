#include <stdio.h>
#include <cuda_runtime.h>

#define M 5  // Input Matrix Rows
#define N 5  // Input Matrix Columns
#define MASK_SIZE 3  // Mask (Filter) Size

// CUDA Kernel for 2D Convolution
__global__ void convolution2D(int *input, int *mask, int *output, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int halfMask = MASK_SIZE / 2;
    int sum = 0;

    if (row >= halfMask && row < m - halfMask && col >= halfMask && col < n - halfMask) {
        for (int i = -halfMask; i <= halfMask; i++) {
            for (int j = -halfMask; j <= halfMask; j++) {
                int inputRow = row + i;
                int inputCol = col + j;
                int maskRow = i + halfMask;
                int maskCol = j + halfMask;
                sum += input[inputRow * n + inputCol] * mask[maskRow * MASK_SIZE + maskCol];
            }
        }
        output[row * n + col] = sum;
    }
}

// Function to print a matrix
void printMatrix(int *M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", M[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int size_input = M * N * sizeof(int);
    int size_mask = MASK_SIZE * MASK_SIZE * sizeof(int);
    int size_output = M * N * sizeof(int);

    // Example Input Matrix
    int input[M * N] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    // Example 3x3 Mask (Filter)
    int mask[MASK_SIZE * MASK_SIZE] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    int output[M * N] = {0};  // Output matrix

    int *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_mask, size_mask);
    cudaMalloc(&d_output, size_output);

    cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);

    // Define 2D Grid and Block size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch Kernel
    convolution2D<<<numBlocks, threadsPerBlock>>>(d_input, d_mask, d_output, M, N);

    cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost);

    printf("Input Matrix:\n");
    printMatrix(input, M, N);

    printf("\nMask (Filter):\n");
    printMatrix(mask, MASK_SIZE, MASK_SIZE);

    printf("\nOutput Matrix after Convolution:\n");
    printMatrix(output, M, N);

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);

    return 0;
}


/*
#include <stdio.h>
#include <cuda_runtime.h>

// Define MAX size for matrix and mask
#define MAX_M 10  
#define MAX_N 10  
#define MAX_MASK 5  

// CUDA Kernel for 2D Convolution
__global__ void convolution2D(int *input, int *mask, int *output, int m, int n, int maskSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int halfMask = maskSize / 2;
    int sum = 0;

    if (row >= halfMask && row < m - halfMask && col >= halfMask && col < n - halfMask) {
        for (int i = -halfMask; i <= halfMask; i++) {
            for (int j = -halfMask; j <= halfMask; j++) {
                int inputRow = row + i;
                int inputCol = col + j;
                int maskRow = i + halfMask;
                int maskCol = j + halfMask;
                sum += input[inputRow * n + inputCol] * mask[maskRow * maskSize + maskCol];
            }
        }
        output[row * n + col] = sum;
    }
}

// Function to take matrix input
void inputMatrix(int *matrix, int rows, int cols, const char *name) {
    printf("Enter %s matrix (%d x %d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%d", &matrix[i * cols + j]);
        }
    }
}

// Function to print a matrix
void printMatrix(int *M, int rows, int cols, const char *name) {
    printf("\n%s Matrix:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", M[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int m, n, maskSize;
    
    // Get input size
    printf("Enter number of rows and columns for input matrix: ");
    scanf("%d %d", &m, &n);

    if (m > MAX_M || n > MAX_N) {
        printf("Matrix size exceeds the maximum limit (%dx%d)!\n", MAX_M, MAX_N);
        return -1;
    }

    printf("Enter mask size (must be odd, e.g., 3, 5): ");
    scanf("%d", &maskSize);

    if (maskSize > MAX_MASK || maskSize % 2 == 0) {
        printf("Invalid mask size! It must be an odd number and <= %d\n", MAX_MASK);
        return -1;
    }

    int size_input = m * n * sizeof(int);
    int size_mask = maskSize * maskSize * sizeof(int);
    int size_output = m * n * sizeof(int);

    int *input = (int *)malloc(size_input);
    int *mask = (int *)malloc(size_mask);
    int *output = (int *)malloc(size_output);

    // Take user input for matrices
    inputMatrix(input, m, n, "Input");
    inputMatrix(mask, maskSize, maskSize, "Mask (Filter)");

    int *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_mask, size_mask);
    cudaMalloc(&d_output, size_output);

    cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);

    // Define 2D Grid and Block size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch Kernel
    convolution2D<<<numBlocks, threadsPerBlock>>>(d_input, d_mask, d_output, m, n, maskSize);

    cudaMemcpy(output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Print results
    printMatrix(input, m, n, "Input");
    printMatrix(mask, maskSize, maskSize, "Mask (Filter)");
    printMatrix(output, m, n, "Output after Convolution");

    // Free memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
    free(input);
    free(mask);
    free(output);

    return 0;
}

*/