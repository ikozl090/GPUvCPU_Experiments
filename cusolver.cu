#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <sys/time.h>
#include <lapacke.h>

#include <fstream> // For file operations
#include <iostream> // For standard I/O (optional, for error handling)

// Define general constants 
#define NUM_GPUS 3 // Number of available GPUs 
#define MAX_PRINTABLE_MATRIX_DIM 15 // Max n to print matrix of dim n x n

// Constants for memory magnitudes 
#define KB 1024
#define MB 1048576
#define GB 1073741824

void cusolver(int dim, double times[2]) {

    // Print matrices if below max printable dimension 
    bool print_matrices = false;
    if (dim <= MAX_PRINTABLE_MATRIX_DIM) {
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
            printf("\nError setting the CUDA device: %s\n", cudaGetErrorString(err));
        }

        // Save initial memory before program exacution for all GPUs 
        cudaMemGetInfo(&freeMemBefore[i], &totalMemBefore[i]);

        printf("Amount of free memory in GPU %d before execution is %.4f GB out of %.4f GB total.\n", gpu_id, ((double)freeMemBefore[i])/(1000000000), ((double)totalMemBefore[i])/(1000000000));
    }

    // Linear system size parameters 
    int n = dim; // Matrix A is n x n and vector b is n x 1
    int rows_A = n; 
    int cols_A = n; 
    int lda = rows_A; // leading dimension of array
    int ldb = rows_A; // leading dimension of array
    int nrhs = 1; // Number of right-hand sides (i.e., number of b vectors)

    // Declare pointers for linear system parameters 
    float *A, *d_A, *b, *d_b; 
    long int size_A = sizeof(float) * rows_A * cols_A; 
    long int size_b = sizeof(float) * lda; 
    
    // Allocate memory for matrix and vector 
    printf("\nAllocating memory for A... (%lld bytes / %.2f GB)\n", size_A, ((float) size_A / GB)); 
    A = (float *)malloc(size_A); 
    if (A == 0) {
        printf("malloc failed for A!\n");
    }
    printf("Allocating memory for b... (%lld bytes / %.2f KB)\n", size_b, ((float) size_b / KB)); 
    b = (float *)malloc(size_b); 
    if (b == 0) {
        printf("malloc failed for b!\n");
    }
    printf("Memory allocated successfully.\n");

    // Initialize matrix and vector 
    float max_matrix_val = (float) n; // Make value size proportional to matrix dim to avoid having an unsolvable matrix
    float min_matrix_val = -max_matrix_val;
    for (long int i = 0; i < n; i++){
        for (long int j = 0; j < n; j++){
            A[n * i + j] = (float) rand() / ((float) RAND_MAX + 1) * (max_matrix_val - min_matrix_val) + min_matrix_val; 
        }
        b[i] = (float) rand() / ((float) RAND_MAX +  1) * (max_matrix_val - min_matrix_val) + min_matrix_val; 
    }
    printf("A and b initialized successfully.\n");

    // Print initial matrices if desirable
    if (print_matrices){
        // Print initialized matrix 
        printf("\nA = \n");
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
        printf("\nError setting the CUDA device: %s\n", cudaGetErrorString(err));
    }

    // Initialize start time variables 
    struct timeval start_time, end_time; 
    double run_time;
    gettimeofday(&start_time, NULL); // Get start time 

    // Allocate GPU memory for matrices 
    printf("\nAllocating memory on GPU for A... (%lld bytes / %.2f GB)\n", size_A, ((float) size_A / GB)); 
    cudaMalloc((void **)&d_A, size_A); 
    printf("Allocating memory on GPU for b... (%lld bytes / %.2f KB)\n", size_b, ((float) size_b / KB)); 
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
        fprintf(stderr, "\nLU decomposition failed\n"); 
    }

    // Copy results to CPU 
    float* x = (float*)malloc(rows_A * sizeof(float)); 
    cudaMemcpy(x, d_b, sizeof(float) * rows_A, cudaMemcpyDeviceToHost); 

    // Print results
    if (print_matrices){
        printf("\nx = \n");
        for (int idx = 0; idx < ldb; idx++) {
            printf(" %f ", x[idx]);
            printf("\n");
        }
        printf("\n");
    }

    // Print memory usage results for all GPUs
    printf("\n");
    for (int i = 0; i < NUM_GPUS; i++) {
        // Set the GPU you want to use
        int gpu_id = i; // Replace with the GPU ID you want to use
        cudaError_t err = cudaSetDevice(gpu_id);

        if (err != cudaSuccess) {
            printf("\nError setting the CUDA device: %s\n", cudaGetErrorString(err));
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

    // Finalize and clean up
    cusolverDnDestroy(solver_handle);

    // Get end time 
    gettimeofday(&end_time, NULL);

    // Print running time 
    run_time = (double) (end_time.tv_sec - start_time.tv_sec); 
    run_time += (double) (end_time.tv_usec - start_time.tv_usec)/1000000; 
    printf("\nTotal run time (GPU): %.4f seconds. \n", run_time);

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
        printf("\nThe factorization has a zero diagonal element %d.\n", info);
    }

    // Solve the system Ax = b
    setenv("OPENBLAS_NUM_THREADS", "1", 1); // Only use one thread for CPU computation 
    info = LAPACKE_sgetrs(LAPACK_COL_MAJOR, 'N', n, nrhs, A, lda, ipiv, b, ldb);

    if (info > 0) {
        printf("\nThe solve operation failed %d.\n", info);
    }

    // Print results
    if (print_matrices){
        printf("\nx = \n");
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
    printf("\nTotal run time (CPU): %.4f seconds. \n", run_time_cpu);

    // Finish run
    printf("Completed CPU Run Successfully!\n");

    // Initialize output 
    times[0] = run_time;
    times[1] = run_time_cpu; 
}

int main(int argc, char *argv[]) {
    // Ensure matrix dimension was given
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_number> (Please enter matrix dimensions n for n x n matrix)\n", argv[0]);
        return 1;
    }

    int max_dim = atoi(argv[1]);  // Convert the argument to an integer
    double times[2]; // Initialize time array

    // Open the file in append mode to prevent overwriting existing content
    std::ofstream outFile("data.csv", std::ios::app);

    // Check if the file is open
    if (!outFile.is_open()) {
        printf("Failed to open the file for writing.");
        return 1; // Exit with an error code
    }

    // Write initial row 
    outFile << "Dimension" << "," << "GPU" << "," << "CPU" << std::endl;

    for (int i = 1000; i <= max_dim; i += 1000) {
        
        int mat_dim = i; // Get matrix dimension 

        // Compute times 
        cusolver(mat_dim, times);

        // Write time values as a comma-separated row
        outFile << mat_dim << "," << times[0] << "," << times[1] << std::endl; 
    }
    

    // Close the file
    outFile.close();

    return 0;
}