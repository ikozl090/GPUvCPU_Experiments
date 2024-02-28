#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverMg.h> 
#include "cusolverMg_utils.h"
#include "cusolver_utils.h"
#include <cublas_v2.h>
#include <sys/time.h>

// Define general constants 
#define NUM_GPUS 3 // Number of available GPUs 
#define MAX_PRINTABLE_MATRIX_DIM 15 // Max n to print matrix of dim n x n

// Constants for memory magnitudes 
#define KB 1024
#define MB 1048576
#define GB 1073741824

int main(int argc, char *argv[]) {

    /*********************************
        Initialize Linear System 
    **********************************/

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

    // Linear system size parameters 
    int n = input; // Matrix A is n x n and vector b is n x 1
    int rows_A = n; 
    int cols_A = n; 
    int IA = 1; 
    int JA = 1; 
    int IB = 1; 
    int JB = 1; 
    int T_A = 256;
    int T_B = 1;  
    int lda = rows_A; // leading dimension of array
    int ldb = rows_A; // leading dimension of array
    int info = 0; 

    // Declare pointers for linear system parameters 
    float *A, *b, *x; 
    int *IPIV;
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
           Solve System on GPUs  
    **********************************/

    // Initialize cuSOLVERMG handle variable
    cusolverMgHandle_t cusolverMgHandle;

    // Initialize system descriptors 
    cudaLibMgMatrixDesc_t descr_A;
    cudaLibMgMatrixDesc_t descr_b;
    cudaLibMgGrid_t grid_A;
    cudaLibMgGrid_t grid_b;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    // Initialize workspace parameters 
    int64_t lwork_getrf = 0;
    int64_t lwork_getrs = 0;
    int64_t lwork = 0; /* workspace: number of elements per device */

    printf("\nCreating environment...\n"); 

    // Create cuSOLVERMG environment
    cusolverStatus_t createStatus = cusolverMgCreate(&cusolverMgHandle); 

    // Verify successfuly handle creation 
    if (createStatus == CUSOLVER_STATUS_ALLOC_FAILED) {
        printf("\nThe resource could not be allocated.\n"); 
        return 1; 
    }

    printf("\nInitializing devices...\n"); 

    // Initialize devices 
    int devices[NUM_GPUS]; 
    for (int i = 0; i < NUM_GPUS; i++) { 
        devices[i] = i; // Each logical device points to corresponding physical device 
    }

    // Register logical devices with cuSOLVERMG handle 
    cusolverStatus_t deviceSelectStatus = cusolverMgDeviceSelect(cusolverMgHandle, NUM_GPUS, devices); 

    // Verify successfuly device selection 
    if (deviceSelectStatus != CUSOLVER_STATUS_SUCCESS) {
        printf("\nThe resource could not be allocated.\n"); 
         if (deviceSelectStatus == CUSOLVER_STATUS_INVALID_VALUE) {
            printf("\nnbDevices must be greater than zero, and less or equal to 32.\n"); 
         } 
         
         if (deviceSelectStatus == CUSOLVER_STATUS_ALLOC_FAILED) {
            printf("\nThe resources could not be allocated.\n"); 
         } 
         
         if (deviceSelectStatus == CUSOLVER_STATUS_INTERNAL_ERROR) {
            printf("\nInternal error occured when setting internal steams and events.\n"); 
         }

        return 1; // Exit process 
    }

    printf("\nEnabling peer access...\n"); 

    // Enabling peer access 
    enablePeerAccess(NUM_GPUS, devices);

    printf("\nCreate matrix descriptors for A and b... \n"); 

    // Create matrix grids 
    cusolverStatus_t grid_status_A = cusolverMgCreateDeviceGrid(&grid_A, 1, NUM_GPUS, devices, mapping); 
    cusolverStatus_t grid_status_b = cusolverMgCreateDeviceGrid(&grid_b, 1, NUM_GPUS, devices, mapping); 

    // Verify device grid creation 
    if (grid_status_A != CUSOLVER_STATUS_SUCCESS || grid_status_b != CUSOLVER_STATUS_SUCCESS) {
        printf("\nFailed to create grid for A or b.\n");
    }

    // Create matrix descriptors 
    cusolverStatus_t descr_status_A = cusolverMgCreateMatrixDesc(&descr_A, n, n, n, T_A, CUDA_R_32F, grid_A);
    cusolverStatus_t descr_status_b = cusolverMgCreateMatrixDesc(&descr_b, n, n, n, T_B, CUDA_R_32F, grid_b);

    // Verify device grid creation 
    if (descr_status_A != CUSOLVER_STATUS_SUCCESS || descr_status_b != CUSOLVER_STATUS_SUCCESS) {
        printf("\nFailed to create matrix descriptor for A or b.\n");
    }

    // Initialize device pointers 
    float *array_d_A[NUM_GPUS]; 
    float *array_d_b[NUM_GPUS]; 
    int *array_d_IPIV[NUM_GPUS]; 
    float *array_d_work[NUM_GPUS]; 

    /* A := 0 */
    createMat<float>(NUM_GPUS, devices, n, /* number of columns of global A */
                         T_A,              /* number of columns per column tile */
                         lda,              /* leading dimension of local A */
                         array_d_A);

    /* b := 0 */
    createMat<float>(NUM_GPUS, devices, 1, /* number of columns of global A */
                         T_B,              /* number of columns per column tile */
                         ldb,              /* leading dimension of local A */
                         array_d_b);

    /* IPIV := 0, IPIV is consistent with A */
    createMat<int>(NUM_GPUS, devices, n, /* number of columns of global IPIV */
                   T_A,                  /* number of columns per column tile */
                   1,                    /* leading dimension of local IPIV */
                   array_d_IPIV);

    // Copy data to devices 
    memcpyH2D<float>(NUM_GPUS, devices, n, n,
                         /* input */
                         A, lda,
                         /* output */
                         n,                /* number of columns of global A */
                         T_A,              /* number of columns per column tile */
                         lda,              /* leading dimension of local A */
                         array_d_A, /* host pointer array of dimension nbGpus */
                         IA, JA);

    memcpyH2D<float>(NUM_GPUS, devices, n, 1,
                         /* input */
                         b, ldb,
                         /* output */
                         1,                /* number of columns of global A */
                         T_B,              /* number of columns per column tile */
                         ldb,              /* leading dimension of local A */
                         array_d_b, /* host pointer array of dimension nbGpus */
                         IB, JB);

    // Compute needed buffer size for LU factorization 
    cusolverStatus_t buffer_factorization_status = cusolverMgGetrf_bufferSize(
        cusolverMgHandle, n, n, reinterpret_cast<void **>(array_d_A), IA, JA, 
        descr_A, array_d_IPIV, CUDA_R_32F, &lwork_getrf);
    // Compute needed buffer size for system solver 
    cusolverMgGetrs_bufferSize(
        cusolverMgHandle, CUBLAS_OP_N, n, 1, /* NRHS */
        reinterpret_cast<void **>(array_d_A), IA, JA, descr_A, array_d_IPIV,
        reinterpret_cast<void **>(array_d_b), IB, JB, descr_b, CUDA_R_32F, &lwork_getrs);

    // Compute workspace size using maximum required 
    lwork = max(lwork_getrf, lwork_getrs);
    printf("\tAllocate device workspace, lwork = %lld \n", static_cast<long long>(lwork));

    /* array_d_work[j] points to device workspace of device j */
    workspaceAlloc(NUM_GPUS, devices,
                   sizeof(float) * lwork, /* number of bytes per device */
                   reinterpret_cast<void **>(array_d_work));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    CUSOLVER_CHECK(
        cusolverMgGetrf(cusolverMgHandle, n, n, reinterpret_cast<void **>(array_d_A), IA, JA,
                        descr_A, array_d_IPIV, CUDA_R_32F,
                        reinterpret_cast<void **>(array_d_work), lwork, &info /* host */
                        ));
    
    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if A is singular */
    if (0 > info) {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    CUSOLVER_CHECK(cusolverMgGetrs(cusolverMgHandle, CUBLAS_OP_N, n, 1, /* NRHS */
                                   reinterpret_cast<void **>(array_d_A), IA, JA, descr_A,
                                   array_d_IPIV, reinterpret_cast<void **>(array_d_b),
                                   IB, JB, descr_b ,CUDA_R_32F,
                                   reinterpret_cast<void **>(array_d_work), lwork,
                                   &info /* host */
                                   ));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if parameters are valid */
    if (0 > info) {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    memcpyD2H<float>(NUM_GPUS, devices, n, 1,
                         /* input */
                         1,   /* number of columns of global B */
                         T_B, /* number of columns per column tile */
                         ldb, /* leading dimension of local B */
                         array_d_b, IB, JB,
                         /* output */
                         x, /* N-by-1 */
                         ldb);

    /* IPIV is consistent with A, use JA and T_A */
    memcpyD2H<int>(NUM_GPUS, devices, 1, n,
                   /* input */
                   n,   /* number of columns of global IPIV */
                   T_A, /* number of columns per column tile */
                   1,   /* leading dimension of local IPIV */
                   array_d_IPIV, 1, JA,
                   /* output */
                   IPIV, /* 1-by-N */
                   1);

    printf("\nDestroying environment...\n"); 

    // Destroy cuSOLVERMG environment 
    cusolverStatus_t destroyStatus = cusolverMgDestroy(cusolverMgHandle);

    // Verify successful handle deletion 
    if (destroyStatus != CUSOLVER_STATUS_SUCCESS) {
        printf("\nThe resource could not be shut down.\n"); 
        return 1; 
    }

    return 0;
}