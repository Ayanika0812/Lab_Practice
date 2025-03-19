#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 5  // Image Width
#define HEIGHT 5 // Image Height
#define MASK_SIZE 3 // Emboss Mask Size

// CUDA Kernel for Emboss Effect
__global__ void embossKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int mask[MASK_SIZE][MASK_SIZE] = {
        {-2, -1,  0},
        {-1,  1,  1},
        { 0,  1,  2}
    };

    int halfMask = MASK_SIZE / 2;
    int sum = 0;

    if (x >= halfMask && x < width - halfMask && y >= halfMask && y < height - halfMask) {
        for (int i = -halfMask; i <= halfMask; i++) {
            for (int j = -halfMask; j <= halfMask; j++) {
                int pixelValue = input[(y + i) * width + (x + j)];
                sum += pixelValue * mask[i + halfMask][j + halfMask];
            }
        }

        sum = min(max(sum + 128, 0), 255); // Normalize to [0,255]
        output[y * width + x] = sum;
    }
}

// Function to print image matrix
void printImage(unsigned char *image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%3d ", image[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int size = WIDTH * HEIGHT * sizeof(unsigned char);

    // Example Grayscale Image (5x5)
    unsigned char input[WIDTH * HEIGHT] = {
        10,  20,  30,  40,  50,
        60,  70,  80,  90, 100,
        110, 120, 130, 140, 150,
        160, 170, 180, 190, 200,
        210, 220, 230, 240, 250
    };

    unsigned char output[WIDTH * HEIGHT] = {0}; // Output image

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Define 2D Grid and Block Size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch Emboss Kernel
    embossKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, WIDTH, HEIGHT);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Original Image:\n");
    printImage(input, WIDTH, HEIGHT);

    printf("\nEmbossed Image:\n");
    printImage(output, WIDTH, HEIGHT);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}


/*
#include <stdio.h>
#include <cuda_runtime.h>

#define MAX_M 10  // Maximum rows
#define MAX_N 10  // Maximum columns
#define MAX_K 5   // Maximum filter size (must be odd)

// CUDA Kernel for 2D Convolution
__global__ void convolution2D(int *input, int *mask, int *output, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int half_k = k / 2;

    if (row < m && col < n) {
        int sum = 0;
        for (int i = -half_k; i <= half_k; i++) {
            for (int j = -half_k; j <= half_k; j++) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < m && c >= 0 && c < n) {  // Boundary check
                    sum += input[r * n + c] * mask[(i + half_k) * k + (j + half_k)];
                }
            }
        }
        output[row * n + col] = sum;
    }
}

// Function to take user input for a matrix
void inputMatrix(int *matrix, int rows, int cols, const char *name) {
    printf("Enter values for %s (%d x %d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%d", &matrix[i * cols + j]);
        }
    }
}

// Function to print a matrix
void printMatrix(int *matrix, int rows, int cols, const char *name) {
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int m, n, k;
    
    // Get input matrix size
    printf("Enter number of rows and columns for input matrix: ");
    scanf("%d %d", &m, &n);

    // Get filter size (must be odd)
    printf("Enter mask size (must be odd, e.g., 3, 5): ");
    scanf("%d", &k);
    if (k % 2 == 0 || k > MAX_K) {
        printf("Invalid mask size! Must be an odd number <= %d.\n", MAX_K);
        return 1;
    }

    int sizeInput = m * n * sizeof(int);
    int sizeMask = k * k * sizeof(int);
    int sizeOutput = m * n * sizeof(int);

    int *h_input = (int *)malloc(sizeInput);
    int *h_mask = (int *)malloc(sizeMask);
    int *h_output = (int *)malloc(sizeOutput);

    // Take user input
    inputMatrix(h_input, m, n, "Input Matrix");
    inputMatrix(h_mask, k, k, "Mask (Filter)");

    // Allocate device memory
    int *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, sizeInput);
    cudaMalloc(&d_mask, sizeMask);
    cudaMalloc(&d_output, sizeOutput);

    // Copy data to device
    cudaMemcpy(d_input, h_input, sizeInput, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeMask, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((n + 15) / 16, (m + 15) / 16);

    // Launch the kernel
    convolution2D<<<gridSize, blockSize>>>(d_input, d_mask, d_output, m, n, k);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeOutput, cudaMemcpyDeviceToHost);

    // Print results
    printMatrix(h_input, m, n, "Input Matrix");
    printMatrix(h_mask, k, k, "Mask (Filter)");
    printMatrix(h_output, m, n, "Output after Convolution");

    // Free memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
    free(h_input);
    free(h_mask);
    free(h_output);

    return 0;
}

*/