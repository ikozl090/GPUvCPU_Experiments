#!/bin/bash 

/opt/rocm/bin/hipcc -I/opt/rocm/include hipsolver.cpp -lrocblas -lrocsolver -o int2

./int2 "$@"