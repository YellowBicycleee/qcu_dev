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
void fullDslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int dagger_flag) {}
void cg_inverter(void* b_vector, void* x_vector, void *gauge, QcuParam *param){}
void loadQcuGauge(void* gauge, QcuParam *param){}




void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity) {
  // getDeviceInfo();
  // parity ---- invert_flag

  // cloverDslashOneRound(fermion_out, fermion_in, gauge, param, 0);
  // cloverDslashOneRound(fermion_out, fermion_in, gauge, param, parity);
  // fullCloverDslashOneRound(fermion_out, fermion_in, gauge, param, 0);
  // wilsonDslashOneRound(fermion_out, fermion_in, gauge, param, parity);
  // callWilsonDslash(fermion_out, fermion_in, gauge, param, parity, 0);

  // callWilsonDslash(fermion_out, fermion_in, gauge, param, parity, 0);

  qcu::callWilsonDslashNaive(fermion_out, fermion_in, gauge, param, parity, 0);
}