import torch 
import cupy 

# Define size of matrix and vector such that A is N x N and b is N
N = 25000

# Define pytorch matrix and vector for linear system 
A = 9 * torch.rand(N,N) + 1
b = 9 * torch.rand(N) + 1

# Send linear system to GPU 
A = A.to("cuda:0")
b = b.to("cuda:0")

# Solve linear system 
x = torch.linalg.solve(A,b) 

print("System Solved!") 
