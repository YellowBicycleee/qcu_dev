#include "qcu_wilson_dslash.cuh"
#include "kernel/qcu_wilson_dslash_naive.cuh"
#include <chrono>
#include <cstdio>

namespace qcu {
  extern int proc_size[Nd];

  void WilsonDslash::calculateDslash() {

  }


  void WilsonDslash::calculateDslashNaive() {

    double flag = (daggerFlag_ == 0) ? 1.0 : -1.0;

    int half_vol = Lx_ * Ly_ * Lz_ * Lt_ >> 1;
    int block_size = BLOCK_SIZE;
    int grid_size = (half_vol + block_size - 1) / block_size;
    dim3 gridDim(grid_size);
    dim3 blockDim(block_size);
  
    qcuCudaDeviceSynchronize();
  
    // mpi_comm->preDslash(dslashParam_->fermion_in, parity, invert_flag);
  
    auto start = std::chrono::high_resolution_clock::now();

    void *args[] = {&gauge_, &fermionIn_, &fermionOut_, &Lx_, &Ly_, &Lz_, &Lt_, \
                    &parity_, &proc_size[0], &proc_size[1], &proc_size[2], \
                    &proc_size[3], &flag};
  
    checkCudaErrors(cudaLaunchKernel((void *)mpiDslashNaive, gridDim, blockDim, args));

    qcuCudaDeviceSynchronize();

    // boundary calculate
    // mpi_comm->postDslash(dslashParam_->fermion_out, parity, invert_flag);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("naive total time: (without malloc free memcpy) : %.9lf sec, block size = %d\n", double(duration) / 1e9, block_size);
  }


  void callWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag) {
    printf("Naive Wilson Dslash...\n");
    DslashParam dslash_param(param, parity, dagger_flag, fermion_in, fermion_out, gauge);
    WilsonDslash dslash_solver(dslash_param);
    dslash_solver.calculateDslashNaive();
  }

  void callFullWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag, double kappa) {

    int Lx = param->lattice_size[0];
    int Ly = param->lattice_size[1];
    int Lz = param->lattice_size[2];
    int Lt = param->lattice_size[3];
    int half_vol = Lx / 2 * Ly * Lz * Lt;
    void* diag_fermion_in = static_cast<void*>(static_cast<Complex*>(fermion_in) + parity * half_vol * Ns * Nc);
    void* non_diag_fermion_in = static_cast<void*>(static_cast<Complex*>(fermion_in) + (1-parity) * half_vol * Ns * Nc);


    DslashParam dslash_param(param, parity, dagger_flag, non_diag_fermion_in, fermion_out, gauge);
    WilsonDslash dslash_solver(dslash_param);
    dslash_solver.calculateDslashNaive();


    // dst = src - kappa dst
    int block_size = BLOCK_SIZE;
    int grid_size = (half_vol + block_size - 1) / block_size;
    mpiDslashNaiveTail<<<grid_size, block_size>>>(gauge, diag_fermion_in, fermion_out, Lx, Ly, Lz, Lt, parity, kappa);
    qcuCudaDeviceSynchronize();
  }
};
