This repository contains correctness tests and performance benchmarks for symmetry preserving algorithms for symmetric tensor contractions, which are described in

http://dx.doi.org/10.3929/ethz-a-010345741

The matlab directory contains implementations of symmetry preserving algorithms (for batches of) matrix-vector products, rank-2 symmetric updates, and symmetrized products of symmetric matrices.

The python directory contains prototype implementations of algorithms for AB+BA where A and B are symmstric and each elementwise oepration is a matrix-multiplication. It also contains a script to test the sensitivity of error for mat-vec relative to condition number of the matrix

Specifically,

* matlab/test_symv.m - given list of matrix dimensions (ns) tests relative error of symmetry preserving algorithm for mat-vec relative to direct evaluation approach, for random input.
* matlab/test_symv2.m - given list of matrix dimensions (ns) tests relative error of symmetry preserving algorithm for mat-vec relative to direct evaluation approach, for input that is designed to maximize error for the symmetry preserving algorithm.
* matlab/test_sysysy.m - given list of matrix dimensions (ns) tests relative error of symmetry preserving algorithm for mat-mat product relative to direct evaluation approach, for random input.
* matlab/test_symv2.m - given list of matrix dimensions (ns) tests relative error of symmetry preserving algorithm for mat-mat product relative to direct evaluation approach, for input that is designed to maximize error for the symmetry preserving algorithm.
* matlab/test_syr2.m - given list of matrix dimensions (ns) tests relative error of symmetry preserving algorithm for symmetrized outer product relative to direct evaluation approach, for random input.
* matlab/test_syr2k.m - given list of matrix dimensions (ns) tests relative error of symmetry preserving algorithm for a sum of k symmetrized outer products relative to direct evaluation approach, for random input.

The c directory contains tests for higher-order versions of symmetry preserving algorithms and a benchmrk for the AB+BA product where each element of the symetric A and B matrices. is a matrix.

The correctness tests relative Cyclops Tensor Framework https://github.com/cyclops-community/ctf and are currently (as of Oct 1, 2019) also includes in the CTF repo under studies.

Specifically,

* c/fast_tensor_ctr.cxx - uses CTF to test symmetry preseving algorithm for any n,s,t,v.
* c/fast_sy_as_as_tensor_ctr.cxx - uses CTF to test symmetry preseving algorithm for any n,s,t,v with antisymmetric tensors as input.
* c/fast_as_as_sy_tensor_ctr.cxx - uses CTF to test symmetry preseving algorithm for any n,s,t,v with an antisymmetric tensor as input and anitsymmetric output.
* c/s1t1v1_blk_seq.cxx - uses C/BLAS to perform AB+BA with elementwise mat-mat products sequentially (build via Makefile).
* c/s1t1v1_blk_omp.cxx - uses C/BLAS to perform AB+BA with elementwise mat-mat products using OpenMP threading over blocks (build via Makefile).
