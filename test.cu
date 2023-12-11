#include <stdio.h> 
#include <cuda.h> 
#include <stdlib.h>
#include <cusolverDn.h>

const int N = 256; // Matrix dim is N x N

__global__ void matrixAdd(int* a, int* b, int* c)
{ 
    // Use corresponding block indices to parse matrix 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    // Sum up matrix elements 
    if (i < N && j < N)
    {
        c[i*N + j] = a[i*N + j] + b[i*N + j];
    }
}

int main() 
{
    int a[N][N], b[N][N], c[N][N]; // Allocate memory for matrices 
    int *d_a, *d_b, *d_c; // Device memory pointers 

    // initialize matrices here 
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            a[i][j] = rand() % 100; 
            b[i][j] = rand() % 100; 
        }
    }

    // Allocate memory on the GPU 
    cudaMalloc((void**)&d_a, N * N * sizeof(int)); 
    cudaMalloc((void**)&d_b, N * N * sizeof(int)); 
    cudaMalloc((void**)&d_c, N * N * sizeof(int)); 

    // Copy matrices to GPU
    cudaMemcpy(d_a, &a, N * N * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, &b, N * N * sizeof(int), cudaMemcpyHostToDevice); 

    // Launch matrixAdd kernel on GPU with N x N block 
    dim3 threadsPerBlock(1,1);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1)/threadsPerBlock.x, (N + threadsPerBlock.y - 1)/threadsPerBlock.y);
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a,d_b,d_c); 

    // Copy results back to memory 
    cudaMemcpy(&c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost); 

    // Memory cleanup 
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 

    printf("Operation done successfully!\n");

    return 0; 
}
