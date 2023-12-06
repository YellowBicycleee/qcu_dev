#pragma once


#include "qcu_dslash_base.cuh"

namespace qcu {

  class WilsonDslash : public Dslash {
  public:
    WilsonDslash(DslashParam& param) : Dslash(param){}
    virtual void calculateDslash();
    virtual void calculateDslashNaive();
  };

  void callWilsonDslash(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);

  void callWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag);

  void callFullWilsonDslashNaive(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity, int dagger_flag = 0, double kappa = 0.125);
}