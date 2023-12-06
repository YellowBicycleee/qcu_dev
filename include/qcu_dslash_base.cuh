#pragma once
#include "qcu.h"


namespace qcu {


  enum DslashType {
    KERNEL_DSLASH = 0,
    FULL_WILSON = 1,
    FULL_CLOVER = 2,
  };

  enum DslashDaggerFlag {
    NODAGGER = 0,
    DAGGER = 1,
  };

  // host class
  struct DslashParam {
    int Lx;
    int Ly;
    int Lz;
    int Lt;
    int parity;

    void* fermion_in;
    void* fermion_out;
    void* gauge;

    DslashParam(void* p_fermion_in, void* p_fermion_out, void* p_gauge, QcuParam* p_qcu_param, int p_parity) \
      : fermion_in(p_fermion_in), fermion_out(p_fermion_out), gauge(p_gauge),  \
        Lx(p_qcu_param->lattice_size[0]), Ly(p_qcu_param->lattice_size[1]), \
        Lz(p_qcu_param->lattice_size[2]), Lt(p_qcu_param->lattice_size[3]), parity(p_parity){}
    DslashParam(const DslashParam& rhs) : fermion_in(rhs.fermion_in), fermion_out(rhs.fermion_out), gauge(rhs.gauge), Lx(rhs.Lx), Ly(rhs.Ly), Lz(rhs.Lz), Lt(rhs.Lt), parity(rhs.parity) {}
    DslashParam& operator= (const DslashParam& rhs) {
      fermion_in = rhs.fermion_in;
      fermion_out = rhs.fermion_out;
      gauge = rhs.gauge;
      Lx = rhs.Lx;
      Ly = rhs.Ly;
      Lz = rhs.Lz;
      Lt = rhs.Lt;
      parity = rhs.parity;
      return *this;
    }
  };

  // host class
  class Dslash {
  protected:
    DslashParam *dslashParam_;
    int Lx_;
    int Ly_;
    int Lz_;
    int Lt_;
    int parity_;
  public:
    Dslash(DslashParam& param) : dslashParam_(&param){}
    virtual void calculateDslash(int invert_flag) = 0;
  };

};