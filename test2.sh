#!/bin/bash 

nvcc -I/usr/include/openblas -o cusolver test2_cusolver.cu -lcusolver -llapacke -llapack -lblas
./cusolver "$@"

