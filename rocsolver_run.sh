#!/bin/bash 

/opt/rocm/bin/hipcc -I/opt/rocm/include rocsolver.cpp -lrocblas -lrocsolver -o int2

./int2 "$@"