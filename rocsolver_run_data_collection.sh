#!/bin/bash 

/opt/rocm/bin/hipcc -I/opt/rocm/include rocsolver_collect_data.cpp -lrocblas -lrocsolver -o int2

./int2 "$@"