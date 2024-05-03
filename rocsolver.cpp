/////////////////////////////
// example.cpp source code //
/////////////////////////////

#include <algorithm> // for std::min
#include <stddef.h>  // for size_t
// #include <vector>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations

#include <sstream>
#include <iomanip>
#include <string>

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

int main(int argc, char *argv[]) {

  // Ensure matrix dimension was given
  if (argc != 2) {
      std::fprintf(stderr, "Usage: %s <input_number> (Please enter matrix dimensions N for N x N matrix)\n", argv[0]);
      return 1;
  }

  int input = atoi(argv[1]);  // Convert the argument to an integer

  // Print available devices
  int device_count; 
  HIP_ERROR(hipGetDeviceCount(&device_count)); 
  // if (count_error != 0) {
  //     std::printf("Device count failed with error %d\n", count_error); 
  // }
  std::printf("\nNumber of compute capable devices = %d\n", device_count); 

  hipDeviceProp_t device_prop; 
  std::string device_name;
  size_t total_mem_bytes; 
  // hipError_t device_prop_error;  
  for (int device_id = 0; device_id < device_count; device_id++) {
    HIP_ERROR(hipGetDeviceProperties(&device_prop, device_id)); 
    // if (device_prop_error != 0) {
    //   std::printf("Device property reading failed with error %d\n", device_prop_error); 
    // }

    device_name = device_prop.name; 
    total_mem_bytes = device_prop.totalGlobalMem;
    std::printf("Device %d: %s | Available memory = %s\n", device_id, device_name.c_str(), getMemoryString(total_mem_bytes).c_str()); 
  }

  // Check initial device ID 
  int initial_device_id; 
  HIP_ERROR(hipGetDevice(&initial_device_id)); 
  // if (id_error != 0) {
  //   std::printf("Device ID reading failed with error %d\n", id_error); 
  // }
  std::printf("\nInital device ID = %d\n", initial_device_id); 

  // Set to desired device
  int set_device_id = 1; 
  std::printf("Setting to device %d\n", set_device_id); 
  HIP_ERROR(hipSetDevice(set_device_id)); 
  // if (id_set_error != 0) {
  //   std::printf("Device ID setting failed with error %d\n", id_set_error); 
  // }

  // Confirm new device ID 
  int current_device_id; 
  HIP_ERROR(hipGetDevice(&current_device_id)); 
  // if (new_id_error != 0) {
  //   std::printf("Device ID reading failed with error %d\n", new_id_error); 
  // }
  std::printf("New device ID = %d\n", current_device_id);

  // return 0;

  //optional call to rocblas_initialize
  // rocblas_initialize();

  // note the order, call hipSetDevice before hipStreamCreate
  hipStream_t stream;
  HIP_ERROR(hipStreamCreate(&stream));

  rocblas_handle handle;
  // rocblas_status create_status = rocblas_create_handle(&handle);
  ROCBLAS_STATUS(rocblas_create_handle(&handle));
  // return 0;
  //rocblas_status workspace_create_status = rocblas_set_workspace(handle, 0, 0);
  // std::printf("Create Status: %u\n", create_status); 

  ROCBLAS_STATUS(rocblas_set_stream(handle, stream));

  size_t memory; 
  ROCBLAS_STATUS(rocblas_get_device_memory_size(handle, &memory));
  std::printf("Memory = %s\n", getMemoryString(memory).c_str()); 

  // Check if memory is managed automatically
  bool mem_managed = &rocblas_is_managing_device_memory; 

  std::printf("Automatic memory management status: %d\n", mem_managed); 

  // return 0; 

  rocblas_int N = input;
  rocblas_int lda = N;
  rocblas_int ldb = lda; 
  rocblas_int info = -777; 

  size_t size_A = size_t(lda) * N;          // the size of the array for the matrix
  size_t size_B = size_t(ldb);              // the size of the array for the vector
  size_t size_X = size_t(N);                // the size of the array for the output vector
  size_t size_piv = size_t(N); // the size of array for the Householder scalars

  std::printf("Number of matrix elements = %zu\n", size_A);
  std::printf("Matrix size = %s\n", getMemoryString(sizeof(data_type) * size_A).c_str());
  std::printf("Number of vector elements = %zu\n", size_B);
  std::printf("Vector size = %s\n", getMemoryString(sizeof(data_type) * size_B).c_str());

  // here is where you would initialize M, N and lda with desired values

  // return 0; 

  // size_t req_memory_size; 
  // rocblas_start_device_memory_size_query(handle);
  // rocsolver_dgetrf(handle, N, N, nullptr, lda, nullptr, nullptr);
  // rocsolver_dgetrs(handle, rocblas_operation_none, N, 1, nullptr, lda, nullptr, nullptr, ldb); 
  // // hipMalloc(nullptr, sizeof(data_type)*size_A);
  // // hipMalloc(nullptr, sizeof(rocblas_int)*size_piv);
  // // hipMemcpy(nullptr, nullptr, sizeof(data_type)*size_A, hipMemcpyHostToDevice);
  // rocblas_stop_device_memory_size_query(handle, &req_memory_size);

  // std::printf("Required memory: %.2f MB\n", (req_memory_size/1e6)); 

  // return 0;  

  // hipDeviceProp_t device_prop; 
  // hipError_t device_prop_error = hipGetDeviceProperties(&device_prop, device_id); 
  // std::string device_name = device_prop.name; 
  // std::printf("Device properties error status = %d\n", device_prop_error); 
  // std::printf("Current device name: %s\n", device_name.c_str()); 

  // return 0;

  // size_t memory_size;
  // rocblas_start_device_memory_size_query(handle);
  // rocsolver_dgetrf(handle, 1024, 1024, nullptr, lda, nullptr, nullptr);
  // rocsolver_dgetrs(handle, rocblas_operation_none, 1024, 1, nullptr, lda, nullptr, nullptr, lda);
  // rocblas_stop_device_memory_size_query(handle, &memory_size);

  // std::printf("%zu\n", memory_size);

  // data_type hA[size_A]; // This causes segmentation fault error
  data_type *hA, *hB, *hX;      // This doesn't
  hA = (data_type *) malloc(sizeof(data_type) * size_A);
  hB = (data_type *) malloc(sizeof(data_type) * size_B); 
  hX = (data_type *) malloc(sizeof(data_type) * size_X);

  rocblas_int *hIpiv; // creates array for householder scalars in CPU
  hIpiv = (rocblas_int *) malloc(sizeof(rocblas_int) * size_piv);

  // Initialize hA with random values 
  // initialize_random_matrix(hA, N, M); 
  initialize_random_linsys(hA, hB, N); 

  // Define device pointers
  data_type *dA, *dB;  
  rocblas_int *dIpiv;

  HIP_ERROR(hipMalloc(&dA, sizeof(data_type)*size_A));      // allocates memory for matrix in GPU
  HIP_ERROR(hipMalloc(&dB, sizeof(data_type)*size_B));      // allocates memory for vector in GPU
  HIP_ERROR(hipMalloc(&dIpiv, sizeof(rocblas_int)*size_piv)); // allocates memory for scalars in GPU

  // rocblas_set_workspace(handle, &dA, sizeof(data_type)*size_A);
  // rocblas_set_workspace(handle, &dIpiv, sizeof(rocblas_int)*size_piv);
  // These cause memory allocastion error in rocsolver_dgetrf

  // std::printf("\nmallocA Status = %d\n", mallocA_hiperror); 
  // std::printf("mallocB Status = %d\n", mallocB_hiperror); 
  // std::printf("mallocIpiv Status = %d\n", mallocIpiv_hiperror); 
  // std::printf("Device Matrix Address = %zu\n", &dA);

  // return 0; 

  // here is where you would initialize matrix A (array hA) with input data
  // note: matrices must be stored in column major format,
  //       i.e. entry (i,j) should be accessed by hA[i + j*lda]

  // copy data to GPU

  // Allows setting amount of memory directly 
  // hipStream_t stream;
  // rocblas_get_stream(handle, &stream);
  // hipStreamSynchronize(stream);
  // rocblas_set_device_memory_size(handle, 1e9);

  HIP_ERROR(hipMemcpy(dA, hA, sizeof(data_type)*size_A, hipMemcpyHostToDevice));
  HIP_ERROR(hipMemcpy(dB, hB, sizeof(data_type)*size_B, hipMemcpyHostToDevice));
  // std::printf("dA memory copy error = %d\n", memcpy_dA_hiperror);
  // std::printf("dB memory copy error = %d\n", memcpy_dB_hiperror);

  // return 0; 

  ROCBLAS_STATUS(rocblas_get_device_memory_size(handle, &memory));
  std::printf("Memory = %s\n", getMemoryString(memory).c_str()); 

  // return 0;

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
  ROCBLAS_STATUS(rocsolver_dgetrs(handle, rocblas_operation_none, N, 1, dA, lda, dIpiv, dB, ldb));

  // return 0; 

  // copy the results back to CPU
  // hipError_t memcpy_hA_hiperror = hipMemcpy(hA, dA, sizeof(data_type)*size_A, hipMemcpyDeviceToHost);
  HIP_ERROR(hipMemcpy(hX, dB, sizeof(data_type)*size_X, hipMemcpyDeviceToHost));
  // hipError_t memcpy_Ipiv_hiperror = hipMemcpy(hIpiv, dIpiv, sizeof(rocblas_int)*size_piv, hipMemcpyDeviceToHost);
  // std::printf("hA memory copy error = %d\n", memcpy_hA_hiperror);
  // std::printf("hX memory copy error = %d\n", memcpy_hX_hiperror);
  // std::printf("hIpiv memory copy error = %d\n", memcpy_Ipiv_hiperror);

  // the results are now in hA and hIpiv, so you can use them here

  // hipError_t hipfree_dA_hiperror = hipFree(dA);                        // de-allocate GPU memory
  // hipError_t hipfree_dB_hiperror = hipFree(dB);                        // de-allocate GPU memory
  // hipError_t hipfree_Ipiv_hiperror = hipFree(dIpiv);                   // de-allocate GPU memory
  HIP_ERROR(hipFree(dA));
  HIP_ERROR(hipFree(dB)); 
  HIP_ERROR(hipFree(dIpiv)); 

  // Synchronize the non-default stream before destroying it
  HIP_ERROR(hipStreamSynchronize(stream));

  HIP_ERROR(hipStreamDestroy(stream));

  ROCBLAS_STATUS(rocblas_destroy_handle(handle));     // destroy handle

  // Compute residual 

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
  std::printf("|b - x|_inf = %E\n", max_bx_diff);
  std::printf("|x|_inf = %E\n", x_nrm_inf);
  std::printf("|b|_inf = %E\n", b_nrm_inf);
  std::printf("|A|_inf = %E\n", A_nrm_inf);
  /* relative error is around machine zero  */
  /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
  std::printf("|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);

}