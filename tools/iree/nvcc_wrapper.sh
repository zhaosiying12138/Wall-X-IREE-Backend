#!/bin/bash
# Host compiler shim used by setup.py on this RTX 5090 Laptop / CUDA 13.3 /
# gcc-15 host. torch 2.10 headers are accepted by clang-21 but rejected by
# the system GCC 15 template diagnostics.
exec /usr/local/cuda-13.3/bin/nvcc -ccbin clang-21 "$@"
