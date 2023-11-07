#include "qcu_complex.cuh"
#include "qcu_point.cuh"
#include "qcu_macro.cuh"


#define INCLUDE_COMPUTATION
using namespace qcu;


static __device__ __forceinline__ void reconstructSU3(Complex *su3)
{
  su3[6] = (su3[1] * su3[5] - su3[2] * su3[4]).conj();
  su3[7] = (su3[2] * su3[3] - su3[0] * su3[5]).conj();
  su3[8] = (su3[0] * su3[4] - su3[1] * su3[3]).conj();
}

static __device__ __forceinline__ void loadGauge(Complex* u_local, void* gauge_ptr, int direction, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* u = p.getPointGauge(static_cast<Complex*>(gauge_ptr), direction, Lx, Ly, Lz, Lt);
  for (int i = 0; i < (Nc - 1) * Nc; i++) {
    u_local[i] = u[i];
  }
  reconstructSU3(u_local);
}
static __device__ __forceinline__ void loadVector(Complex* src_local, void* fermion_in, const Point& p, int Lx, int Ly, int Lz, int Lt) {
  Complex* src = p.getPointVector(static_cast<Complex *>(fermion_in), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    src_local[i] = src[i];
  }
}



__global__ void mpiDslashNaive(void *gauge, void *fermion_in, void *fermion_out, int Lx, int Ly, int Lz, int Lt, int parity, int grid_x, int grid_y, int grid_z, int grid_t, double flag_param) {
  assert(parity == 0 || parity == 1);
  Lx >>= 1;
  int half_Lx = Lx >> 1;

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread_id / (Lz * Ly * half_Lx);
  int z = thread_id % (Lz * Ly * half_Lx) / (Ly * half_Lx);
  int y = thread_id % (Ly * half_Lx) / half_Lx;
  int x = thread_id % half_Lx;

  int coord_boundary;
  double flag = flag_param;


  Point p(x, y, z, t, parity);
  Point move_point;
  Complex u_local[Nc * Nc];   // for GPU
  Complex src_local[Ns * Nc]; // for GPU
  Complex dst_local[Ns * Nc]; // for GPU
  // Complex temp;
  Complex temp1;
  Complex temp2;
  int eo = (y+z+t) & 0x01;

  for (int i = 0; i < Ns * Nc; i++) {
    dst_local[i].clear2Zero();
  }

  // \mu = 1
  loadGauge(u_local, gauge, X_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 0, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  // x front    x == Lx-1 && parity != eo
  coord_boundary = (grid_x > 1 && x == Lx-1 && parity != eo) ? Lx-1 : Lx;
  if (x < coord_boundary) {

#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        temp1 += (src_local[0 * Nc + j] - src_local[3 * Nc + j].multiply_i() * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[2 * Nc + j].multiply_i() * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1.multiply_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2.multiply_i() * flag;
    }
#endif
  }
  // x back   x==0 && parity == eo
  move_point = p.move(BACK, 0, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, X_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);;

  coord_boundary = (grid_x > 1 && x==0 && parity == eo) ? 1 : 0;
  if (x >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[3 * Nc + j].multiply_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj

        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[2 * Nc + j].multiply_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[3 * Nc + i] += temp1.multiply_minus_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[2 * Nc + i] += temp2.multiply_minus_i() * flag;
    }
#endif
  }

  // \mu = 2
  // y front
  loadGauge(u_local, gauge, Y_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 1, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_y > 1) ? Ly-1 : Ly;
  if (y < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
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
#endif
  }

  // y back
  move_point = p.move(BACK, 1, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Y_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);


  coord_boundary = (grid_y > 1) ? 1 : 0;
  if (y >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
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
#endif
  }

  // \mu = 3
  // z front
  loadGauge(u_local, gauge, Z_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 2, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);
  coord_boundary = (grid_z > 1) ? Lz-1 : Lz;
  if (z < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] - src_local[2 * Nc + j].multiply_i() * flag) * u_local[i * Nc + j];
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] + src_local[3 * Nc + j].multiply_i() * flag) * u_local[i * Nc + j];
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1.multiply_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2.multiply_minus_i() * flag;
    }
#endif
  }

  // z back
  move_point = p.move(BACK, 2, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, Z_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_z > 1) ? 1 : 0;
  if (z >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
#pragma unroll
    for (int i = 0; i < Nc; i++) {
      temp1.clear2Zero();
      temp2.clear2Zero();
#pragma unroll
      for (int j = 0; j < Nc; j++) {
        // first row vector with col vector
        temp1 += (src_local[0 * Nc + j] + src_local[2 * Nc + j].multiply_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
        // second row vector with col vector
        temp2 += (src_local[1 * Nc + j] - src_local[3 * Nc + j].multiply_i() * flag) *
              u_local[j * Nc + i].conj(); // transpose and conj
      }
      dst_local[0 * Nc + i] += temp1;
      dst_local[2 * Nc + i] += temp1.multiply_minus_i() * flag;
      dst_local[1 * Nc + i] += temp2;
      dst_local[3 * Nc + i] += temp2.multiply_i() * flag;
    }
#endif
  }

  // t: front
  // loadGauge(u_local, gauge, 3, p, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, p, Lx, Ly, Lz, Lt);
  move_point = p.move(FRONT, 3, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? Lt-1 : Lt;
  if (t < coord_boundary) {
#ifdef INCLUDE_COMPUTATION
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
#endif
  }
  // t: back
  move_point = p.move(BACK, 3, Lx, Ly, Lz, Lt);
  loadGauge(u_local, gauge, T_DIRECTION, move_point, Lx, Ly, Lz, Lt);
  loadVector(src_local, fermion_in, move_point, Lx, Ly, Lz, Lt);

  coord_boundary = (grid_t > 1) ? 1 : 0;
  if (t >= coord_boundary) {
#ifdef INCLUDE_COMPUTATION
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
#endif
  }

  Complex* dst_global = p.getPointVector(static_cast<Complex *>(fermion_out), Lx, Ly, Lz, Lt);
  for (int i = 0; i < Ns * Nc; i++) {
    dst_global[i] = dst_local[i];
  }
  
}