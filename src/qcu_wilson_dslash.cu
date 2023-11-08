#include "qcu_wilson_dslash.cuh"
#include "kernel/qcu_wilson_dslash_naive.cuh"
#include <chrono>
namespace qcu {

  void WilsonDslash::calculateDslash(int invert_flag) {

  }


  void WilsonDslash::calculateDslashNaive(int dagger_flag) {
    int grid_x = 1, grid_y = 1, grid_z = 1, grid_t = 1;

    int Lx = dslashParam_->Lx;
    int Ly = dslashParam_->Ly;
    int Lz = dslashParam_->Lz;
    int Lt = dslashParam_->Lt;
    int parity = dslashParam_->parity;
    double flag = (dagger_flag == 0) ? 1.0 : -1.0;

    int half_vol = Lx * Ly * Lz * Lt >> 1;
    int block_size = BLOCK_SIZE;
    int grid_size = (half_vol + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(grid_size);
    dim3 blockDim(block_size);
  
    checkCudaErrors(cudaDeviceSynchronize());
  
    // mpi_comm->preDslash(dslashParam_->fermion_in, parity, invert_flag);
  
    auto start = std::chrono::high_resolution_clock::now();
    void *args[] = {&dslashParam_->gauge, &dslashParam_->fermion_in, &dslashParam_->fermion_out, &Lx, &Ly, &Lz, &Lt, &parity, &grid_x, &grid_y, &grid_z, &grid_t, &flag};
  
    checkCudaErrors(cudaLaunchKernel((void *)mpiDslashNaive, gridDim, blockDim, args));
  
    checkCudaErrors(cudaDeviceSynchronize());
    // boundary calculate
    // mpi_comm->postDslash(dslashParam_->fermion_out, parity, invert_flag);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("naive total time: (without malloc free memcpy) : %.9lf sec, block size = %d\n", double(duration) / 1e9, block_size);
  }


  void callWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag) {
    DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
    WilsonDslash dslash_solver(dslash_param);
    dslash_solver.calculateDslashNaive(dagger_flag);
  }

  void callFullWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag, double kappa) {
    DslashParam dslash_param(fermion_in, fermion_out, gauge, param, parity);
    WilsonDslash dslash_solver(dslash_param);
    dslash_solver.calculateDslashNaive(dagger_flag);


    // dst = src - kappa dst
    int Lx = param->lattice_size[0];
    int Ly = param->lattice_size[1];
    int Lz = param->lattice_size[2];
    int Lt = param->lattice_size[3];
    int half_vol = Lx / 2 * Ly * Lz * Lt;
    int block_size = BLOCK_SIZE;
    int grid_size = (half_vol + block_size - 1) / block_size;
    // mpiDslashNaiveTail(void *gauge, void *fermion_in, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, double kappa) 
    mpiDslashNaiveTail<<<grid_size, block_size>>>(gauge, fermion_in, fermion_out, Lx, Ly, Lz, Lt, parity, kappa);
    checkCudaErrors(cudaDeviceSynchronize());
  }
};
