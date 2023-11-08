#pragma once

#include "kernel/qcu_public_kernel.cuh"
#include "qcu_macro.cuh"
#include "qcu_complex.cuh"
#include "qcu_point.cuh"

static __device__ __forceinline__ void loadGaugeCoalesced(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int sub_Lx, int Ly, int Lz, int Lt) {
  Complex* start_ptr = p.getCoalescedGaugeAddr (gauge_ptr, direction, sub_Lx, Ly, Lz, Lt);
  int sub_vol = sub_Lx * Ly * Lz * Lt;
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = *start_ptr;
    start_ptr += sub_vol;
  }
  reconstructSU3(u_local);
}

static __device__ __forceinline__ void loadVectorCoalesced(Complex* src_local, void* fermion_in, const Point& p, int half_Lx, int Ly, int Lz, int Lt) {
  Complex* start_ptr = p.getCoalescedVectorAddr (fermion_in, half_Lx, Ly, Lz, Lt);
  int sub_vol = half_Lx * Ly * Lz * Lt;

  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = *start_ptr;
    start_ptr += sub_vol;
  }
}

static __device__ __forceinline__ void storeVectorCoalesced(Complex* dst_local, void* fermion_out, const Point& p, int half_Lx, int Ly, int Lz, int Lt) {
  Complex* start_ptr = p.getCoalescedVectorAddr (fermion_out, half_Lx, Ly, Lz, Lt);
  int sub_vol = half_Lx * Ly * Lz * Lt;

  for (int i = 0; i < Ns * Nc; i++) {
    *start_ptr = dst_local[i];
    start_ptr += sub_vol;
  }
}


static __global__ void mpiDslashCoalesce(void *gauge, void *fermion_in, void *fermion_out,int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, double flag_param) {
  assert(parity == 0 || parity == 1);
  Lx >>= 1;

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread_id / (Lz * Ly * Lx);
  int z = thread_id % (Lz * Ly * Lx) / (Ly * Lx);
  int y = thread_id % (Ly * Lx) / Lx;
  int x = thread_id % Lx;

  int coord_boundary;
  double flag = flag_param;


  Point p(x, y, z, t, parity);
  Point move_point;
  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  Complex temp1;
  Complex temp2;
  int eo = (y+z+t) & 0x01;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  loadGaugeCoalesced(u_local, gauge, X_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx-1 && parity != eo) ? Lx-1 : Lx;
  if (x < coord_boundary) {
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1.multipy_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2.multipy_i() * flag;
    }
  }

  // x back   x==0 && parity == eo
  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, X_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_x > 1 && x==0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj

        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1.multipy_minus_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2.multipy_minus_i() * flag;
    }
  }


  // \mu = 2
  // y front
  loadGaugeCoalesced(u_local, gauge, Y_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? Ly-1 : Ly;
  if (y < coord_boundary) {
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j] *  flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += -temp2 * flag;
    }
  }

  // y back
  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, Y_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);


  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += -temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2 * flag;
    }
  }

  // \mu = 3
  // z front
  loadGaugeCoalesced(u_local, gauge, Z_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz-1 : Lz;
  if (z < coord_boundary) {

#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j].multipy_i() * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1.multipy_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2.multipy_minus_i() * flag;
    }
  }

  // z back
  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, Z_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j].multipy_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1.multipy_minus_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2.multipy_i() * flag;
    }
  }

  // t: front
  loadGaugeCoalesced(u_local, gauge, T_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt-1 : Lt;
  if (t < coord_boundary) {
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j] * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j] * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += -temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += -temp2 * flag;
    }
  }
  // t: back
  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGaugeCoalesced(u_local, gauge, T_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVectorCoalesced(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j] * flag) * u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1 * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2 * flag;
    }
  }

  // store result
  storeVectorCoalesced(dst_local, fermion_out, p, Lx, Ly, Lz, Lt);
}