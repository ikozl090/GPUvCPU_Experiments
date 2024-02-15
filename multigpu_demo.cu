#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverMg.h> 
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


    printf("\nCreate matrix descriptors for A and b... \n"); 

    // // Create matrix grids 
    // cusolverStatus_t grid_status_A = cusolverMgCreateDeviceGrid(&grid_A, 1, NUM_GPUS, devices, mapping); 
    // cusolverStatus_t grid_status_b = cusolverMgCreateDeviceGrid(&grid_b, 1, NUM_GPUS, devices, mapping); 

    // // Create matrix descriptors 
    // cusolverStatus_t descr_status_A = cusolverMgCreateMatrixDesc(&descr_A, n, n, n, T_A, float, grid_A);
    // cusolverStatus_t descr_status_b = cusolverMgCreateMatrixDesc(&descr_b, n, n, n, T_B, float, grid_b);




    printf("\nDestroying environment...\n"); 

    // Destroy cuSOLVERMG environment 
    cusolverStatus_t destroyStatus = cusolverMgDestroy(cusolverMgHandle);

    // Verify successfuly handle deletion 
    if (destroyStatus != CUSOLVER_STATUS_SUCCESS) {
        printf("\nThe resource could not be shut down.\n"); 
        return 1; 
    }

    return 0;
}