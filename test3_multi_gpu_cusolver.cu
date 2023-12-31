#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <sys/time.h>
#include <lapacke.h>

#define NUM_GPUS 3
#define MAX_PRINTABLE_MATRIX_DIM 15

int main(int argc, char *argv[]) {

    // Ensure matrix dimension was given
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_number> (Please enter matrix dimensions n for n x n matrix)\n", argv[0]);
        return 1;
    }

    int input = atoi(argv[1]);  // Convert the argument to an integer

    // Print matrices if below max printable dimension 
    bool print_matrices = false;
    if (input <= MAX_PRINTABLE_MATRIX_DIM) {
        print_matrices = true; 
    }

    // Variables to keep track of memory usage 
    size_t freeMemBefore[NUM_GPUS], totalMemBefore[NUM_GPUS], freeMemAfter[NUM_GPUS], totalMemAfter[NUM_GPUS];

    // Display initial and free memory for all GPUs
    for (int i = 0; i < NUM_GPUS; i++) {
        // Set the GPU you want to use
        int gpu_id = i; // Replace with the GPU ID you want to use
        cudaError_t err = cudaSetDevice(gpu_id);

        if (err != cudaSuccess) {
            printf("Error setting the CUDA device: %s", cudaGetErrorString(err));
            return 1;
        }

        // Save initial memory before program exacution for all GPUs 
        cudaMemGetInfo(&freeMemBefore[i], &totalMemBefore[i]);

        printf("Amout of free memory in GPU %d before execution is %.4f GB out of %.4f GB total.\n", gpu_id, ((double)freeMemBefore[i])/(1000000000), ((double)totalMemBefore[i])/(1000000000));
    }

    // Linear system size parameters 
    int n = input; // Matrix A is n x n and vector b is n x 1
    int rows_A = n; 
    int cols_A = n; 
    int lda = rows_A; // leading dimension of array
    int ldb = rows_A; // leading dimension of array
    int nrhs = 1; // Number of right-hand sides (i.e., number of b vectors)

    // Allocate memory for matrix and vector 
    float *A, *d_A, *b, *d_b; 
    int size_A = sizeof(float) * rows_A * cols_A; 
    int size_b = sizeof(float) * lda; 
    A = (float *)malloc(size_A); 
    b = (float *)malloc(size_b); 

    // Initialize matrix and vector 
    float max_matrix_val = 10;
    float min_matrix_val = -max_matrix_val;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[n * i + j] = (float) rand() / ((float) RAND_MAX + 1) * (max_matrix_val - min_matrix_val) + min_matrix_val; 
        }
        b[i] = (float) rand() / ((float) RAND_MAX +  1) * (max_matrix_val - min_matrix_val) + min_matrix_val; 
    }

    // Print initial matrices if desirable
    if (print_matrices){
        // Print initialized matrix 
        printf("A = \n");
        for (int i = 0; i < rows_A; i++){
            for (int j = 0; j < cols_A; j++){
                printf(" %f ", A[n * i + j]);
            }
            printf("\n");
        } 

        // Print initialized vector 
        printf("b = \n");
        for (int i = 0; i < rows_A; i++){
            printf(" %f ", b[i]);
            printf("\n");
        } 
    }

    /*********************************
        Perform Operations on GPU  
    **********************************/

    // Set device to first GPU
    cudaError_t err = cudaSetDevice(0);

    if (err != cudaSuccess) {
        printf("Error setting the CUDA device: %s", cudaGetErrorString(err));
        return 1;
    }

    // Initialize start time variables 
    struct timeval start_time, end_time; 
    double run_time;
    gettimeofday(&start_time, NULL); // Get start time 

    // Allocate GPU memory for matrices 
    cudaMalloc((void **)&d_A, size_A); 
    cudaMalloc((void **)&d_b, size_b); 

    // Initialize and create cuSolver handler 
    cusolverDnHandle_t solver_handle; 
    cusolverDnCreate(&solver_handle); 

    // Transfer data from host to device (GPU) 
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice); 

    // Initialize variable for operation and buffer info
    int *devIpiv, *devInfo; // pivot and status information pointers 
    float *Workspace; // Buffer used during LU decomposition  
    int lwork = 0; // Size of workspace buffer used for operation 

    // Compute buffer size required for LU decomposition 
    cusolverDnSgetrf_bufferSize(solver_handle, rows_A, cols_A, d_A, lda, &lwork);
    cudaMalloc((void **)&devInfo, sizeof(int)); 
    cudaMalloc((void **)&devIpiv, sizeof(int) * rows_A); 
    cudaMalloc((void **)&Workspace, sizeof(float) *lwork);

    // Decompose system into LU matrices 
    cusolverDnSgetrf(solver_handle, rows_A, cols_A, d_A, lda, Workspace, devIpiv, devInfo); 

    // Solve system in LU decomposed form 
    cusolverDnSgetrs(solver_handle, CUBLAS_OP_N, n, nrhs, d_A, lda, devIpiv, d_b, lda, devInfo); 
    
    // Check devInfo to ensure cuSOLVER routine went well 
    int devInfo_h = 0; // dev info hat 
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost); // Save devInfo from GPU to devInfo_h on CPU
    if (devInfo_h != 0) {
        fprintf(stderr, "LU decomposition failed\n"); 
    }

    // Copy results to CPU 
    float* x = (float*)malloc(rows_A * sizeof(float)); 
    cudaMemcpy(x, d_b, sizeof(float) * rows_A, cudaMemcpyDeviceToHost); 

    // Print results
    if (print_matrices){
        printf("x = \n");
        for (int idx = 0; idx < ldb; idx++) {
            printf(" %f ", x[idx]);
            printf("\n");
        }
        printf("\n");
    }

    // Print memory usage results for all GPUs
    for (int i = 0; i < NUM_GPUS; i++) {
        // Set the GPU you want to use
        int gpu_id = i; // Replace with the GPU ID you want to use
        cudaError_t err = cudaSetDevice(gpu_id);

        if (err != cudaSuccess) {
            printf("Error setting the CUDA device: %s", cudaGetErrorString(err));
            return 1;
        }

        // Get memory after solver execution for all GPUs
        cudaMemGetInfo(&freeMemAfter[i], &totalMemAfter[i]);

        printf("Memory used by cuSOLVER function for GPU %d: %.4f GB out of %.4f GB total.\n", gpu_id, ((double)(freeMemBefore[i] - freeMemAfter[i]))/1000000000, ((double)totalMemBefore[i])/1000000000);
    }

    // Free up memory 
    cudaFree(Workspace);
    cudaFree(devIpiv); 
    cudaFree(devInfo); 
    cudaFree(d_A); 
    cudaFree(d_b); 
    free(x); 

    // Get end time 
    gettimeofday(&end_time, NULL);

    // Print running time 
    run_time = (double) (end_time.tv_sec - start_time.tv_sec); 
    run_time += (double) (end_time.tv_usec - start_time.tv_usec)/1000000; 
    printf("Total run tim (GPU): %.4f seconds. \n", run_time);

    // Finish run 
    printf("Completed GPU Run Successfully!\n");

    /*********************************
        Perform Operations on CPU  
    **********************************/

    // Initialize variable for matrix operations 
    int ipiv[n]; // Variable for keeping track of pivot indices 
    int info; // To keep track of operation status 
    
    // Initialize start time variables 
    struct timeval start_time_cpu, end_time_cpu; 
    double run_time_cpu;
    gettimeofday(&start_time_cpu, NULL); // Get start time 

    // Perform LU decomposition of A
    setenv("OPENBLAS_NUM_THREADS", "1", 1); // Only use one thread for CPU computation 
    info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, A, lda, ipiv);

    if (info > 0) {
        printf("The factorization has a zero diagonal element %d.\n", info);
        return -1;
    }

    // Solve the system Ax = b
    setenv("OPENBLAS_NUM_THREADS", "1", 1); // Only use one thread for CPU computation 
    info = LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, nrhs, A, lda, ipiv, b, ldb);

    if (info > 0) {
        printf("The solve operation failed %d.\n", info);
        return -1;
    }

    // Print results
    if (print_matrices){
        printf("x = \n");
        for (int idx = 0; idx < ldb; idx++) {
            printf(" %f ", b[idx]);
            printf("\n");
        }
        printf("\n");
    }

    // Get end time 
    gettimeofday(&end_time_cpu, NULL);

    // Print running time 
    run_time_cpu = (double) (end_time_cpu.tv_sec - start_time_cpu.tv_sec); 
    run_time_cpu += (double) (end_time_cpu.tv_usec - start_time_cpu.tv_usec)/1000000; 
    printf("Total run time (CPU): %.4f seconds. \n", run_time_cpu);

    // Finish run
    printf("Completed CPU Run Successfully!\n");

    return 0;
}
