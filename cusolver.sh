#!/bin/bash 

nvcc -I/usr/include/openblas -o cusolver cusolver.cu -lcusolver -llapacke -llapack -lblas
./cusolver "$@"

