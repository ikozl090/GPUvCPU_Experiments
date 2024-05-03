/////////////////////////////
// example.cpp source code //
/////////////////////////////

#include <algorithm> // for std::min
#include <stddef.h>  // for size_t
// #include <vector>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations

// Packages for measuring compute time 
#include <sys/time.h>

// Packages for string operations when printing memory size
#include <sstream>
#include <iomanip>
#include <string>

// Packages for file operations
#include <fstream> // For file operations
#include <iostream> // For standard I/O (optional, for error handling)

// For type checking 
#include <type_traits>

#ifndef IDX2F
#define IDX2F(i, j, lda) ((((j)-1) * (static_cast<size_t>(lda))) + ((i)-1))
#endif /* IDX2F */

#ifndef IDX1F
#define IDX1F(i) ((i)-1)
#endif /* IDX1F */

std::string getMemoryString(size_t mem) { 

  std::ostringstream stream;
  // double memory_num; 
  std::string memory_string;

  if (mem > 1e12) {
    stream << std::fixed << std::setprecision(2) << (((double) mem)/1e12);
    memory_string = stream.str() + " TB"; 
    // memory = std::format("{:.2f} TB", ((double) mem)/1e12);
  } else if (mem > 1e9) {
    // memory_string = std::format("{:.2f} GB", ((double) mem)/1e9);
    stream << std::fixed << std::setprecision(2) << (((double) mem)/1e9);
    memory_string = stream.str() + " GB"; 
  } else if (mem > 1e6) {
    stream << std::fixed << std::setprecision(2) << (((double) mem)/1e6);
    memory_string = stream.str() + " MB"; 
    // memory_string = std::format("{:.2f} MB", ((double) mem)/1e6);
    // memory = std::to_string(mem)
  } else if (mem > 1e3) { 
    stream << std::fixed << std::setprecision(2) << (((double) mem)/1e3);
    memory_string = stream.str() + " kB"; 
    // memory_string = std::format("{:.2f} kB", ((double) mem)/1e3);
  } else {
    memory_string = std::to_string(mem) + " bytes"; 
    // memory_string = std::format("{:d} bits", mem);
  }

  return memory_string;
}

/* compute |x|_inf */
template <typename T> static T vec_nrm_inf(int n, const T *x) {
    T max_nrm = 0.0;
    for (int row = 1; row <= n; row++) {
        T xi = x[IDX1F(row)];
        max_nrm = (max_nrm > fabs(xi)) ? max_nrm : fabs(xi);
    }
    return max_nrm;
}

// Data type to use for linear system 
using data_type = double; 

void ROCBLAS_STATUS(rocblas_status status) {
  switch(status) {
    case rocblas_status_success: 
    std::printf("rocBLAS Status Success! Status = %d\n", status);
    break; 

    case rocblas_status_invalid_size:
    std::printf("Invalid size!\n");
    throw std::runtime_error("rocBLAS error");  
    break;

    case rocblas_status_invalid_pointer: 
    std::printf("Invalid pointer!\n");
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_invalid_handle: 
    std::printf("Invalid handle!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_not_implemented: 
    std::printf("Status not implemented!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_memory_error: 
    std::printf("Failed internal memory allocation, copy or dealloc!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_internal_error: 
    std::printf("Internal error!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_size_query_mismatch: 
    std::printf("Size querry mismatch!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_size_increased: 
    std::printf("Queried device memory size increased!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_size_unchanged: 
    std::printf("Queried device memory size unchanged!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_continue: 
    std::printf("Nothing preventing function to proceed!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break;

    case rocblas_status_invalid_value: 
    std::printf("Passed argument not valid!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break;

    case rocblas_status_check_numerics_fail: 
    std::printf("Will be set if the vector/matrix has a NaN/Infinity/denormal value!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break;

    case rocblas_status_excluded_from_build: 
    std::printf("Function is not available in build, likely a function requiring Tensile built without Tensile!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break;

    case rocblas_status_perf_degraded: 
    std::printf("Performance degraded due to low device memory!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 

    case rocblas_status_arch_mismatch: 
    std::printf("The function requires a feature absent from the device architecture!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break;

    default: 
    std::printf("Unknown failure!\n"); 
    throw std::runtime_error("rocBLAS error");  
    break; 
  }
}

void HIP_ERROR(hipError_t error) {                                                                                                                                                  
  if (error != hipSuccess) {    
    const char *error_name; 
    error_name = hipGetErrorName(error);  
    std::printf("HIP error %d at %s:%d\n", error, __FILE__, __LINE__);    
    std::printf("Error name: %s\n", error_name);
    throw std::runtime_error("HIP error");                                                
  }                                                                      
}

void initialize_random_linsys(data_type *A, data_type *B, rocblas_int N, int max_val = 1){
  // Initilize matrix 
  data_type max_matrix_val = (data_type) max_val; // Make value size proportional to matrix dim to avoid having an unsolvable matrix
  data_type min_matrix_val = -max_matrix_val;
  for (long int i = 0; i < N; i++){
      for (long int j = 0; j < N; j++){
          A[N * i + j] = (data_type) rand() / ((data_type) RAND_MAX + 1) * (max_matrix_val - min_matrix_val) + min_matrix_val; 
      }
      B[i] = (data_type) rand() / ((data_type) RAND_MAX +  1) * (max_matrix_val - min_matrix_val) + min_matrix_val; 
  }
  printf("\nA and b initialized successfully.\n");
}

double rocsolver(int dim) {

    int input = dim;  // Convert the argument to an integer

    // Print available devices
    int device_count; 
    HIP_ERROR(hipGetDeviceCount(&device_count)); 
    std::printf("\nNumber of compute capable devices = %d\n", device_count); 

    // Print available GPUs and their respective memory
    hipDeviceProp_t device_prop; 
    std::string device_name;
    size_t total_mem_bytes; 
    for (int device_id = 0; device_id < device_count; device_id++) {
        HIP_ERROR(hipGetDeviceProperties(&device_prop, device_id)); 

        device_name = device_prop.name; 
        total_mem_bytes = device_prop.totalGlobalMem;
        std::printf("Device %d: %s | Available memory = %s\n", device_id, device_name.c_str(), getMemoryString(total_mem_bytes).c_str()); 
    }

    // Check initial device ID 
    int initial_device_id; 
    HIP_ERROR(hipGetDevice(&initial_device_id)); 
    std::printf("\nInital device ID = %d\n", initial_device_id); 

    // Set to desired device (In this case desired device ID = 1)
    int set_device_id = 1; 
    std::printf("Setting to device %d\n", set_device_id); 
    HIP_ERROR(hipSetDevice(set_device_id)); 

    // Confirm new device ID 
    int current_device_id; 
    HIP_ERROR(hipGetDevice(&current_device_id)); 
    std::printf("New device ID = %d\n", current_device_id);

    // Create stream for HIP API
    // Note the order, call hipSetDevice before hipStreamCreate
    hipStream_t stream;
    HIP_ERROR(hipStreamCreate(&stream));

    // Create handle for rocBLAS API
    rocblas_handle handle;
    ROCBLAS_STATUS(rocblas_create_handle(&handle));

    // Link stream with desired debice to handle so rocBLAS uses the same memory as HIP
    ROCBLAS_STATUS(rocblas_set_stream(handle, stream));

    // Print initially available memory for handle
    size_t memory; 
    ROCBLAS_STATUS(rocblas_get_device_memory_size(handle, &memory));
    std::printf("Memory = %s\n", getMemoryString(memory).c_str()); 

    // Check if memory is managed automatically
    bool mem_managed = &rocblas_is_managing_device_memory; 
    std::printf("Automatic memory management status: %d\n", mem_managed); 

    // Matrix and solver parameters 
    rocblas_int N = input;
    rocblas_int lda = N;
    rocblas_int ldb = lda; 
    rocblas_int info = -777; 

    size_t size_A = size_t(lda) * N;          // the size of the array for the matrix
    size_t size_B = size_t(ldb);              // the size of the array for the vector
    size_t size_X = size_t(N);                // the size of the array for the output vector
    size_t size_piv = size_t(N);              // the size of array for the Householder scalars

    std::printf("Number of matrix elements = %zu\n", size_A);
    std::printf("Matrix size = %s\n", getMemoryString(sizeof(data_type) * size_A).c_str());
    std::printf("Number of vector elements = %zu\n", size_B);
    std::printf("Vector size = %s\n", getMemoryString(sizeof(data_type) * size_B).c_str());

    // Initialize pointers to linear system variables on host device (CPU) 
    data_type *hA, *hB, *hX;      // This doesn't
    hA = (data_type *) malloc(sizeof(data_type) * size_A);
    hB = (data_type *) malloc(sizeof(data_type) * size_B); 
    hX = (data_type *) malloc(sizeof(data_type) * size_X);

    rocblas_int *hIpiv; // creates array for householder scalars in CPU
    hIpiv = (rocblas_int *) malloc(sizeof(rocblas_int) * size_piv);

    // Initialize hA and hB with random values 
    initialize_random_linsys(hA, hB, N); 

    // Initialize start time variables 
    struct timeval start_time, end_time; 
    double run_time;
    gettimeofday(&start_time, NULL); // Get start time 

    // Define device (GPU) pointers
    data_type *dA, *dB;  
    rocblas_int *dIpiv;

    // Allocate memory
    HIP_ERROR(hipMalloc(&dA, sizeof(data_type)*size_A));      // allocates memory for matrix in GPU
    HIP_ERROR(hipMalloc(&dB, sizeof(data_type)*size_B));      // allocates memory for vector in GPU
    HIP_ERROR(hipMalloc(&dIpiv, sizeof(rocblas_int)*size_piv)); // allocates memory for scalars in GPU

    // Copy linsys params from host to device (GPU)
    HIP_ERROR(hipMemcpy(dA, hA, sizeof(data_type)*size_A, hipMemcpyHostToDevice));
    HIP_ERROR(hipMemcpy(dB, hB, sizeof(data_type)*size_B, hipMemcpyHostToDevice));

    // Print new handle memory
    ROCBLAS_STATUS(rocblas_get_device_memory_size(handle, &memory));
    std::printf("Memory = %s\n", getMemoryString(memory).c_str()); 

    // compute the PLU factorization on the GPU
    ROCBLAS_STATUS(rocsolver_dgetrf(handle, N, N, dA, lda, dIpiv, &info));

    // info > 0 => factorization can't be found, U is singular, info = 0 => some other internal error 
    if (info > 0) { 
    std::printf("Upper matrix U is singular!\n");
    } else if (info != 0) { 
    std::printf("Unknown error, info = %d but should be 0!\n", info);
    } else if (info == 0) { 
    std::printf("PLU factorization successful!\n");
    }

    // Solve linear system using PLU factorization
    ROCBLAS_STATUS(rocsolver_dgetrs(handle, rocblas_operation_none, N, 1, dA, lda, dIpiv, dB, ldb));

    // copy the results back to CPU
    HIP_ERROR(hipMemcpy(hX, dB, sizeof(data_type)*size_X, hipMemcpyDeviceToHost));

    // Deallocate device memory 
    HIP_ERROR(hipFree(dA));
    HIP_ERROR(hipFree(dB)); 
    HIP_ERROR(hipFree(dIpiv)); 
    
    // Get end time 
    gettimeofday(&end_time, NULL);

    // Print running time 
    run_time = (double) (end_time.tv_sec - start_time.tv_sec); 
    run_time += (double) (end_time.tv_usec - start_time.tv_usec)/1000000; 
    printf("\nTotal run time: %.4f seconds. \n", run_time);

    // Synchronize the non-default stream before destroying it
    HIP_ERROR(hipStreamSynchronize(stream));
    // Destroy stream 
    HIP_ERROR(hipStreamDestroy(stream));
    // Destroy handle
    ROCBLAS_STATUS(rocblas_destroy_handle(handle));

    // Compute and print residual values to test result 

    std::printf("\nMeasure residual error |b - A*x| \n");
    data_type max_err = 0;
    data_type max_bx_diff = 0; 
    for (int row = 1; row <= N; row++) {
        data_type sum = 0.0;
        for (int col = 1; col <= N; col++) {
            data_type Aij = hA[IDX2F(row, col, lda)];
            data_type xj = hX[IDX1F(col)];
            sum += Aij * xj;
        }
        data_type bi = hB[IDX1F(row)];
        data_type err = fabs(bi - sum);
        data_type bx_diff = fabs(bi - hX[IDX1F(row)]); 

        max_bx_diff = (max_bx_diff > bx_diff) ? max_bx_diff : bx_diff; 
        max_err = (max_err > err) ? max_err : err;
    }
    data_type x_nrm_inf = vec_nrm_inf(N, hX);
    data_type b_nrm_inf = vec_nrm_inf(N, hB);

    data_type A_nrm_inf = 4.0;
    data_type rel_err = max_err / (A_nrm_inf * x_nrm_inf + b_nrm_inf);
    std::printf("\n|b - A*x|_inf = %E\n", max_err);
    std::printf("|b - x|_inf = %E\n", max_bx_diff); // Checks if x different from b, i.e. if solver performed any computations
    std::printf("|x|_inf = %E\n", x_nrm_inf);
    std::printf("|b|_inf = %E\n", b_nrm_inf);
    std::printf("|A|_inf = %E\n", A_nrm_inf);
    /* relative error is around machine zero  */
    /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
    std::printf("|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);

    return run_time; 
}

int main(int argc, char *argv[]) {
    // Ensure matrix dimension was given
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <input_number> (Please enter matrix dimensions n for n x n matrix, one for starting dim, one for final)\n", argv[0]);
        return 1;
    }

    int start_dim = atoi(argv[1]); // Convert the argument to an integer
    int max_dim = atoi(argv[2]);  // Convert the argument to an integer
    double time; // Initialize time array

    // Set file name to save data in
    std::string file_name = "rocsolver_data"; 
    if (std::is_same<data_type, double>::value) {
        file_name = file_name + "_dp"; // Indicate data is for double precision run
    } else if (std::is_same<data_type, float>::value) {
        file_name = file_name + "_sp"; // Indicate data is for single precision run
    } else {
        std::printf("Invalid type for linear system values!\n");
        throw std::runtime_error("Invalid type");
    }

    // Open the file in append mode to prevent overwriting existing content
    file_name = file_name + ".csv"; 
    std::ofstream outFile(file_name.c_str(), std::ios::app);

    // Check if the file is open
    if (!outFile.is_open()) {
        printf("Failed to open the file for writing.");
        return 1; // Exit with an error code
    }

    // Write initial row 
    outFile << "Dimension" << "," << "GPU" << std::endl;

    for (int i = start_dim; i <= max_dim; i += 1000) {
        
        int mat_dim = i; // Get matrix dimension 

        // Compute times 
        time = rocsolver(mat_dim);
        std::printf("Finished run with N = %d\n\n", i);

        // Write time values as a comma-separated row
        outFile << mat_dim << "," << time << std::endl; 
    }
    
    // Close the file
    outFile.close();

    return 0;
}