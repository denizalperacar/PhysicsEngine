#ifndef B0AF69A6_577B_4BF9_AF88_BB09112AEE6E
#define B0AF69A6_577B_4BF9_AF88_BB09112AEE6E

#include "../common/common.h"
#include "rotation_dyads.h"

PE_BEGIN

template <typename T>
struct htm_t {
  
  PE_HOST_DEVICE htm_t() = default;
  
  PE_HOST_DEVICE htm_t(const matrix_t<T, 4, 4>& htm) : matrix(htm) {}
  
  PE_HOST_DEVICE htm_t(const matrix_t<T, 3, 3>& dcm, const vector_t<T, 3>& position) {
    from_pdcm(dcm, position);
  }

  PE_HOST_DEVICE htm_t(const quaternion_t<T>& q, const vector_t<T, 3>& position) {
    from_pdcm(q.to_dcm(), position);
  }

  PE_HOST_DEVICE htm_t(vector_t<T, 3> position) {
    set_position(position);
  }

  PE_HOST_DEVICE htm_t(const quaternion_t<T>& q) {
    set_quaternion(q);
  }

  PE_HOST_DEVICE void set_dcm(const matrix_t<T, 3, 3>& dcm) {
    PE_UNROLL
    for (int i = 0; i < 3; ++i) {
      PE_UNROLL
      for (int j = 0; j < 3; ++j) {
        matrix[i][j] = dcm[i][j];
      }
    }
  }

  PE_HOST_DEVICE void set_position(const vector_t<T, 3>& position) {
    PE_UNROLL
    for (int i = 0; i < 3; ++i) {
      matrix[i][3] = position[i];
    }
  }

  PE_HOST_DEVICE void set_quaternion(const quaternion_t<T>& q) {
    set_dcm(q.to_dcm());
  }

  PE_HOST_DEVICE void from_pdcm(const matrix_t<T, 3, 3>& dcm, const vector_t<T, 3>& position) {
    set_dcm(dcm);
    set_position(position);
  }

  PE_HOST_DEVICE void from_pquat(const quaternion_t<T>& q, const vector_t<T, 3>& position) {
    set_quaternion(q);
    set_position(position);
  }

  PE_HOST_DEVICE void from_htm(const matrix_t<T, 4, 4>& htm) {
    matrix = htm;
  }

  PE_HOST_DEVICE void reset() {
    matrix = matrix_t<T, 4, 4>::identity();
  }

  matrix_t<T, 4, 4> matrix = matrix_t<T, 4, 4>::identity();
};

PE_END

#endif /* B0AF69A6_577B_4BF9_AF88_BB09112AEE6E */
