#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

int main() {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

    const int n = 3;  // Size of the matrix
    const int lda = n; // Leading dimension of A
    float A[lda*n] = {  // The matrix A
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0
    };
    float B[n] = {1.0, 2.0, 3.0}; // The vector b

    // Device memory
    float *d_A = NULL;
    float *d_B = NULL;
    int *devIpiv = NULL, *devInfo = NULL;
    int lwork = 0;
    float *d_work = NULL;

    // Step 1: Create cuSolver handle, bind a stream
    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);

    // Step 2: Allocate memory on the device
    cudaMalloc((void **)&d_A, sizeof(float) * lda * n);
    cudaMalloc((void **)&d_B, sizeof(float) * n);
    cudaMalloc((void **)&devIpiv, sizeof(int) * n);
    cudaMalloc((void **)&devInfo, sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * n, cudaMemcpyHostToDevice);

    // Step 3: Query working space of getrf and getrs
    cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, lda, &lwork);
    cudaMalloc((void **)&d_work, sizeof(float) * lwork);

    // Step 4: LU factorization
    cusolverDnSgetrf(cusolverH, n, n, d_A, lda, d_work, devIpiv, devInfo);

    // Step 5: Solve Ax = b
    cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, 1, d_A, lda, devIpiv, d_B, n, devInfo);

    // Copy result back to host
    cudaMemcpy(B, d_B, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // Step 6: Check result
    printf("Solution: \n");
    for (int i = 0; i < n; i++) {
        printf("%f\n", B[i]);
    }

    // Clean up
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (devIpiv) cudaFree(devIpiv);
    if (devInfo) cudaFree(devInfo);
    if (d_work) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream) cudaStreamDestroy(stream);

    printf("Completed Successfully!");

    return 0;
}
