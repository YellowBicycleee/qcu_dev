#include "qcu.h"
#include "qcu_wilson_dslash.cuh"

#include <iostream>
using std::cout;
using std::endl;

// using namespace qcu;

#define qcuPrint() { \
    printf("function %s line %d...\n", __FUNCTION__, __LINE__); \
}


// TODO
void initGridSize(QcuGrid_t* grid, QcuParam* p_param, void* gauge, void* fermion_in, void* fermion_out) {}
void cg_inverter(void* b_vector, void* x_vector, void *gauge, QcuParam *param){}
void loadQcuGauge(void* gauge, QcuParam *param){}




void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {

  int dagger_flag = 0;
  qcu::callWilsonDslashNaive(fermion_out, fermion_in, gauge, param, parity, dagger_flag);
}

void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag, double kappa) {
  // int dagger_flag = 0;
  qcu::callFullWilsonDslashNaive(fermion_out, fermion_in, gauge, param, parity, dagger_flag, kappa);
}