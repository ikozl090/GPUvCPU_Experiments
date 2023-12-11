import torch 
import numpy as np 

# Define size of square matrix given by N x N with vector of size N
N = 1024

# Define a dense square matrix A and the right hand side vector b of the equation Ax = b
A = 9 * np.random.rand(N,N) + 1
b = 9 * np.random.rand(N) + 1

# Convert matrix and vector to CuPy arrays for GPU computation 
A_gpu = cp.asarray(A) 
b_gpu = cp.asarray(b) 

# Solve the linear system Ax = b using cuSOLVER
x_gpu = cp.linalg.solve(A_gpu, b_gpu) 

# Convert the results back to numpy array 
x = cp.asnumpy(x_gpu) 

print("System solved!")
