/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverMg.h>

#include "cusolverMg_utils.h"
#include "cusolver_utils.h"

#include <sys/time.h>

// Constants for memory magnitudes 
#define KB 1024
#define MB (1024 * KB)
#define GB (1024 * MB)

/* compute |x|_inf */
template <typename T> static T vec_nrm_inf(int n, const T *x) {
    T max_nrm = 0.0;
    for (int row = 1; row <= n; row++) {
        T xi = x[IDX1F(row)];
        max_nrm = (max_nrm > fabs(xi)) ? max_nrm : fabs(xi);
    }
    return max_nrm;
}

/* A is 1D laplacian, return A(N:-1:1, :) */
template <typename T> static void gen_1d_laplacian(int N, T *A, int lda) {
    for (int J = 1; J <= N; J++) {
        /* A(J,J) = 2 */
        A[IDX2F(N - J + 1, J, lda)] = 2.0;
        if ((J - 1) >= 1) {
            /* A(J, J-1) = -1*/
            A[IDX2F(N - J + 1, J - 1, lda)] = -1.0;
        }
        if ((J + 1) <= N) {
            /* A(J, J+1) = -1*/
            A[IDX2F(N - J + 1, J + 1, lda)] = -1.0;
        }
    }
}

int main(int argc, char *argv[]) {
    // Ensure matrix dimension was given
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_number> (Please enter matrix dimensions N for N x N matrix)\n", argv[0]);
        return 1;
    }

    int input = atoi(argv[1]);  // Convert the argument to an integer

    cusolverMgHandle_t cusolverH = NULL;

    using data_type = float;

    /* maximum number of GPUs */
    const int MAX_NUM_DEVICES = 16;

    int nbGpus = 0;
    std::vector<int> deviceList(MAX_NUM_DEVICES);

    const int N = input;
    const int IA = 1;
    const int JA = 1;
    const int T_A = 256; /* tile size */
    const int lda = N;

    const int IB = 1;
    const int JB = 1;
    const int T_B = 100; /* tile size of B */
    const int ldb = N;

    int info = 0;

    cudaLibMgMatrixDesc_t descrA;
    cudaLibMgMatrixDesc_t descrB;
    cudaLibMgGrid_t gridA;
    cudaLibMgGrid_t gridB;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    int64_t lwork_getrf = 0;
    int64_t lwork_getrs = 0;
    int64_t lwork = 0; /* workspace: number of elements per device */

    std::printf("Test Random N x N matrix with N = %d\n", N);

    std::printf("Step 1: Create Mg handle and select devices \n");
    CUSOLVER_CHECK(cusolverMgCreate(&cusolverH));

    CUDA_CHECK(cudaGetDeviceCount(&nbGpus));

    nbGpus = (nbGpus < MAX_NUM_DEVICES) ? nbGpus : MAX_NUM_DEVICES;
    std::printf("\tThere are %d GPUs \n", nbGpus);
    for (int j = 0; j < nbGpus; j++) {
        deviceList[j] = j;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, j));
        std::printf("\tDevice %d, %s, cc %d.%d \n", j, prop.name, prop.major, prop.minor);
    }

    CUSOLVER_CHECK(cusolverMgDeviceSelect(cusolverH, nbGpus, deviceList.data()));

    // Variables to keep track of memory usage 
    size_t freeMemBefore[nbGpus], totalMemBefore[nbGpus], freeMemAfter[nbGpus], totalMemAfter[nbGpus];

    int currentDev = 0; /* record current device ID */
    CUDA_CHECK(cudaGetDevice(&currentDev));

    // Display initial free memory for all GPUs
    for (int i = 0; i < nbGpus; i++) {
        // Set the GPU you want to use
        int gpu_id = i; // Replace with the GPU ID you want to use
        CUDA_CHECK(cudaSetDevice(gpu_id));

        // Save initial memory before program exacution for all GPUs 
        cudaMemGetInfo(&freeMemBefore[i], &totalMemBefore[i]);

        std::printf("\tGPU %d, Free Memory: %.4f GB / %.4f GB\n", gpu_id, ((double)freeMemBefore[i])/GB, ((double)totalMemBefore[i])/GB);
    }

    CUDA_CHECK(cudaSetDevice(currentDev));

    std::printf("step 2: Enable peer access.\n");
    enablePeerAccess(nbGpus, deviceList.data());

    std::printf("Step 3: Allocate host memory A \n");
    // std::vector<data_type> A(lda * N, 0);
    // std::vector<data_type> B(ldb, 0);
    // std::vector<data_type> X(ldb, 0);
    // std::vector<int> IPIV(N, 0);
    // data_type A[lda * N];
    // data_type B[ldb];
    // data_type X[ldb];
    // int IPIV[N];

    // Declare pointers for linear system parameters 
    data_type *A, *B, *X; 
    int *IPIV;
    long int size_A = sizeof(data_type) * lda * N; 
    long int size_B = sizeof(data_type) * ldb;
    long int size_X = sizeof(data_type) * N;
    long int size_IPIV = sizeof(data_type) * lda * N;
    
    // Allocate memory for matrix and vector 
    printf("\tAllocating memory for A... (%lld bytes / %.2f GB)\n", size_A, ((float) size_A / GB)); 
    A = (data_type *)malloc(size_A); 
    if (A == 0) {
        printf("\nmalloc failed for A!\n");
    }
    printf("\tAllocating memory for b... (%lld bytes / %.2f KB)\n", size_B, ((float) size_B / KB)); 
    B = (data_type *)malloc(size_B); 
    if (B == 0) {
        printf("\nmalloc failed for b!\n");
    }
    printf("\tAllocating memory for x... (%lld bytes / %.2f KB)\n", size_X, ((float) size_X / KB)); 
    X = (data_type *)malloc(size_X); 
    if (X == 0) {
        printf("\nmalloc failed for x!\n");
    }
    printf("\tAllocating memory for IPIV... (%lld bytes / %.2f GB)\n", size_IPIV, ((float) size_IPIV / GB)); 
    IPIV = (int *)malloc(size_IPIV); 
    if (IPIV == 0) {
        printf("\nmalloc failed for x!\n");
    }
    printf("\tMemory allocated successfully.\n");

    std::printf("Step 4: Prepare random matrix \n");
    // Initialize matrix and vector 
    data_type max_matrix_val = (data_type) N; // Make value size proportional to matrix dim to avoid having an unsolvable matrix
    data_type min_matrix_val = -max_matrix_val;
    for (long int i = 0; i < N; i++){
        for (long int j = 0; j < N; j++){
            A[N * i + j] = (data_type) rand() / ((data_type) RAND_MAX + 1) * (max_matrix_val - min_matrix_val) + min_matrix_val; 
        }
        B[i] = (data_type) rand() / ((data_type) RAND_MAX +  1) * (max_matrix_val - min_matrix_val) + min_matrix_val; 
    }
    printf("A and b initialized successfully.\n");

    // gen_1d_laplacian<data_type>(N, &A[IDX2F(IA, JA, lda)], lda);

#ifdef SHOW_FORMAT
    std::printf("A = matlab base-1\n");
    print_matrix(N, N, A.data(), lda);
#endif

    /* B = ones(N,1) */
    // for (int row = 1; row <= N; row++) {
    //     B[IDX1F(row)] = 1.0;
    // }

#ifdef SHOW_FORMAT
    std::printf("B = matlab base-1\n");
    print_matrix(N, 1, B.data(), ldb, CUBLAS_OP_T);
#endif

    std::printf("Step 5: Create matrix descriptors for A and B \n");

    CUSOLVER_CHECK(cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping));
    CUSOLVER_CHECK(cusolverMgCreateDeviceGrid(&gridB, 1, nbGpus, deviceList.data(), mapping));

    /* (global) A is N-by-N */
    CUSOLVER_CHECK(cusolverMgCreateMatrixDesc(&descrA, N, /* nubmer of rows of (global) A */
                                              N,          /* number of columns of (global) A */
                                              N,          /* number or rows in a tile */
                                              T_A,        /* number of columns in a tile */
                                              traits<data_type>::cuda_data_type, gridA));

    /* (global) B is N-by-1 */
    CUSOLVER_CHECK(cusolverMgCreateMatrixDesc(&descrB, N, /* nubmer of rows of (global) B */
                                              1,          /* number of columns of (global) B */
                                              N,          /* number or rows in a tile */
                                              T_B,        /* number of columns in a tile */
                                              traits<data_type>::cuda_data_type, gridB));

    std::printf("Step 6: Allocate distributed matrices A and B \n");

    std::vector<data_type *> array_d_A(nbGpus, nullptr);
    std::vector<data_type *> array_d_B(nbGpus, nullptr);
    std::vector<int *> array_d_IPIV(nbGpus, nullptr);

    /* A := 0 */
    createMat<data_type>(nbGpus, deviceList.data(), N, /* number of columns of global A */
                         T_A,                          /* number of columns per column tile */
                         lda,                          /* leading dimension of local A */
                         array_d_A.data());

    /* B := 0 */
    createMat<data_type>(nbGpus, deviceList.data(), 1, /* number of columns of global B */
                         T_B,                          /* number of columns per column tile */
                         ldb,                          /* leading dimension of local B */
                         array_d_B.data());

    /* IPIV := 0, IPIV is consistent with A */
    createMat<int>(nbGpus, deviceList.data(), N, /* number of columns of global IPIV */
                   T_A,                          /* number of columns per column tile */
                   1,                            /* leading dimension of local IPIV */
                   array_d_IPIV.data());

    // Initialize start time variables 
    struct timeval start_time, end_time; 
    double run_time;
    std::printf("Begin time measurement...\n");
    gettimeofday(&start_time, NULL); // Get start time 

    std::printf("Step 7: Prepare data on devices \n");
    memcpyH2D<data_type>(nbGpus, deviceList.data(), N, N,
                         /* input */
                         A, lda,
                         /* output */
                         N,                /* number of columns of global A */
                         T_A,              /* number of columns per column tile */
                         lda,              /* leading dimension of local A */
                         array_d_A.data(), /* host pointer array of dimension nbGpus */
                         IA, JA);

    memcpyH2D<data_type>(nbGpus, deviceList.data(), N, 1,
                         /* input */
                         B, ldb,
                         /* output */
                         1,                /* number of columns of global A */
                         T_B,              /* number of columns per column tile */
                         ldb,              /* leading dimension of local A */
                         array_d_B.data(), /* host pointer array of dimension nbGpus */
                         IB, JB);

    std::printf("Step 8: Allocate workspace space \n");
    CUSOLVER_CHECK(cusolverMgGetrf_bufferSize(
        cusolverH, N, N, reinterpret_cast<void **>(array_d_A.data()), IA, /* base-1 */
        JA,                                                               /* base-1 */
        descrA, array_d_IPIV.data(), traits<data_type>::cuda_data_type, &lwork_getrf));

    CUSOLVER_CHECK(cusolverMgGetrs_bufferSize(
        cusolverH, CUBLAS_OP_N, N, 1, /* NRHS */
        reinterpret_cast<void **>(array_d_A.data()), IA, JA, descrA, array_d_IPIV.data(),
        reinterpret_cast<void **>(array_d_B.data()), IB, JB, descrB,
        traits<data_type>::cuda_data_type, &lwork_getrs));

    lwork = std::max(lwork_getrf, lwork_getrs);
    std::printf("\tAllocate device workspace, lwork = %lld \n", static_cast<long long>(lwork));

    std::vector<data_type *> array_d_work(nbGpus, nullptr);

    /* array_d_work[j] points to device workspace of device j */
    workspaceAlloc(nbGpus, deviceList.data(),
                   sizeof(data_type) * lwork, /* number of bytes per device */
                   reinterpret_cast<void **>(array_d_work.data()));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Step 9: Solve A*X = B by GETRF and GETRS \n");
    CUSOLVER_CHECK(
        cusolverMgGetrf(cusolverH, N, N, reinterpret_cast<void **>(array_d_A.data()), IA, JA,
                        descrA, array_d_IPIV.data(), traits<data_type>::cuda_data_type,
                        reinterpret_cast<void **>(array_d_work.data()), lwork, &info /* host */
                        ));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if A is singular */
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    CUSOLVER_CHECK(cusolverMgGetrs(cusolverH, CUBLAS_OP_N, N, 1, /* NRHS */
                                   reinterpret_cast<void **>(array_d_A.data()), IA, JA, descrA,
                                   array_d_IPIV.data(), reinterpret_cast<void **>(array_d_B.data()),
                                   IB, JB, descrB, traits<data_type>::cuda_data_type,
                                   reinterpret_cast<void **>(array_d_work.data()), lwork,
                                   &info /* host */
                                   ));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if parameters are valid */
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    std::printf("Step 10: Retrieve IPIV and solution vector X\n");

    memcpyD2H<data_type>(nbGpus, deviceList.data(), N, 1,
                         /* input */
                         1,   /* number of columns of global B */
                         T_B, /* number of columns per column tile */
                         ldb, /* leading dimension of local B */
                         array_d_B.data(), IB, JB,
                         /* output */
                         X, /* N-by-1 */
                         ldb);

    /* IPIV is consistent with A, use JA and T_A */
    memcpyD2H<int>(nbGpus, deviceList.data(), 1, N,
                   /* input */
                   N,   /* number of columns of global IPIV */
                   T_A, /* number of columns per column tile */
                   1,   /* leading dimension of local IPIV */
                   array_d_IPIV.data(), 1, JA,
                   /* output */
                   IPIV, /* 1-by-N */
                   1);

    std::printf("End time measurement...\n");

    // Get end time 
    gettimeofday(&end_time, NULL);

    // Print running time 
    run_time = (double) (end_time.tv_sec - start_time.tv_sec); 
    run_time += (double) (end_time.tv_usec - start_time.tv_usec)/1000000; 
    std::printf("\nTotal run time (multi-GPU): %.4f seconds. \n", run_time);

    // Display final free memory for all GPUs
    CUDA_CHECK(cudaGetDevice(&currentDev));    
    for (int i = 0; i < nbGpus; i++) {
        // Set the GPU you want to use
        int gpu_id = i; // Replace with the GPU ID you want to use
        CUDA_CHECK(cudaSetDevice(gpu_id));

        // Save initial memory before program exacution for all GPUs 
        cudaMemGetInfo(&freeMemAfter[i], &totalMemAfter[i]);

        std::printf("GPU %d, Memory Used: %.4f GB / %.4f GB\n", gpu_id, ((double)(freeMemBefore[i] - freeMemAfter[i]))/GB, ((double)totalMemBefore[i])/GB);
    }
    CUDA_CHECK(cudaSetDevice(currentDev));

    // Finish run 
    std::printf("Completed GPU Run Successfully!\n");

#ifdef SHOW_FORMAT
    /* X is N-by-1 */
    std::printf("X = matlab base-1\n");
    print_matrix(N, 1, X, ldb, CUBLAS_OP_T);
#endif

#ifdef SHOW_FORMAT
    /* IPIV is 1-by-N */
    std::printf("IPIV = matlab base-1, 1-by-%d matrix\n", N);
    for (int row = 1; row <= N; row++) {
        std::printf("IPIV(%d) = %d \n", row, IPIV[IDX1F(row)]);
    }
#endif

    std::printf("Step 11: Measure residual error |b - A*x| \n");
    data_type max_err = 0;
    for (int row = 1; row <= N; row++) {
        data_type sum = 0.0;
        for (int col = 1; col <= N; col++) {
            data_type Aij = A[IDX2F(row, col, lda)];
            data_type xj = X[IDX1F(col)];
            sum += Aij * xj;
        }
        data_type bi = B[IDX1F(row)];
        data_type err = fabs(bi - sum);

        max_err = (max_err > err) ? max_err : err;
    }
    data_type x_nrm_inf = vec_nrm_inf(N, X);
    data_type b_nrm_inf = vec_nrm_inf(N, B);

    data_type A_nrm_inf = 4.0;
    data_type rel_err = max_err / (A_nrm_inf * x_nrm_inf + b_nrm_inf);
    std::printf("\n|b - A*x|_inf = %E\n", max_err);
    std::printf("|x|_inf = %E\n", x_nrm_inf);
    std::printf("|b|_inf = %E\n", b_nrm_inf);
    std::printf("|A|_inf = %E\n", A_nrm_inf);
    /* relative error is around machine zero  */
    /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
    std::printf("|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);

    std::printf("step 12: Free resources \n");
    destroyMat(nbGpus, deviceList.data(), N, /* number of columns of global A */
               T_A,                          /* number of columns per column tile */
               reinterpret_cast<void **>(array_d_A.data()));
    destroyMat(nbGpus, deviceList.data(), 1, /* number of columns of global B */
               T_B,                          /* number of columns per column tile */
               reinterpret_cast<void **>(array_d_B.data()));
    destroyMat(nbGpus, deviceList.data(), N, /* number of columns of global IPIV */
               T_A,                          /* number of columns per column tile */
               reinterpret_cast<void **>(array_d_IPIV.data()));

    workspaceFree(nbGpus, deviceList.data(), reinterpret_cast<void **>(array_d_work.data()));

    return EXIT_SUCCESS;
}
