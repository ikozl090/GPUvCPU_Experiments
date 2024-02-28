#!/bin/bash 

nvcc -I/usr/include/openblas -o cusolverMg cusolver_MgGetrf_example.cu -lcusolver -llapacke -llapack -lblas -lcusolverMg -diag-suppress 549

./cusolverMg "$@"

