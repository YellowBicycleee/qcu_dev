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
    int daggerFlag;

    void* fermion_in;
    void* fermion_out;
    void* gauge;

    DslashParam(QcuParam* p_qcu_param, int p_parity, int dagger_flag, \
                void* p_fermion_in, void* p_fermion_out, void* p_gauge) \
      : Lx(p_qcu_param->lattice_size[0]), Ly(p_qcu_param->lattice_size[1]), \
        Lz(p_qcu_param->lattice_size[2]), Lt(p_qcu_param->lattice_size[3]), \
        parity(p_parity), daggerFlag(dagger_flag), fermion_in(p_fermion_in), \
        fermion_out(p_fermion_out), gauge(p_gauge) {}

    DslashParam(const DslashParam& rhs) : Lx(rhs.Lx), Ly(rhs.Ly), Lz(rhs.Lz), Lt(rhs.Lt), parity(rhs.parity), daggerFlag(rhs.daggerFlag), fermion_in(rhs.fermion_in), fermion_out(rhs.fermion_out), gauge(rhs.gauge) {}

    DslashParam& operator= (const DslashParam& rhs) {
      Lx = rhs.Lx;
      Ly = rhs.Ly;
      Lz = rhs.Lz;
      Lt = rhs.Lt;
      parity = rhs.parity;
      daggerFlag = rhs.daggerFlag;
      fermion_in = rhs.fermion_in;
      fermion_out = rhs.fermion_out;
      gauge = rhs.gauge;
      return *this;
    }
  };

  // host class
  class Dslash {
  protected:
    int Lx_;
    int Ly_;
    int Lz_;
    int Lt_;
    int parity_;
    int daggerFlag_;
    void* fermionIn_;
    void* fermionOut_;
    void* gauge_;
  public:
    Dslash(const DslashParam& param) : Lx_(param.Lx), Ly_(param.Ly), Lz_(param.Lz),\
            Lt_(param.Lt), parity_(param.parity), daggerFlag_(param.daggerFlag), \
            fermionIn_(param.fermion_in), fermionOut_(param.fermion_out), \
            gauge_(param.gauge) {}
    virtual void calculateDslash() = 0;
  };

};