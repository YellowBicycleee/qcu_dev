#include "qcu.h"
#include "qcu_wilson_dslash.cuh"
#include "qcu_macro.cuh"
#include <cstddef>
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
  qcu::callFullWilsonDslashNaive(fermion_out, fermion_in, gauge, param, parity, dagger_flag, kappa);
}


namespace qcu {
  struct Gauge_t {
    void* ptr;
    bool avail;
    Gauge_t() : ptr(nullptr), avail(false) {}
    Gauge_t(void* p_ptr, bool p_avail) : ptr(p_ptr), avail(p_avail) {}
  };
  int proc_size[Nd];
  int latt_size[Nd];
  // void* qcu_naive_gauge;
  // void* qcu_coalesced_gauge;
  Gauge_t qcu_naive_gauge;
  Gauge_t qcu_coalesced_gauge;
};




__attribute__((constructor)) void init_qcu() {
  qcu::proc_size[0] = 1;
  qcu::proc_size[1] = 1;
  qcu::proc_size[2] = 1;
  qcu::proc_size[3] = 1;
}