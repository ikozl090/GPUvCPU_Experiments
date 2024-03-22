#!/bin/bash 

nvcc -I/usr/include/openblas -o cusolverMg cusolverMg.cu -lcusolver -llapacke -llapack -lblas -lcusolverMg -diag-suppress 549

./cusolverMg "$@"

