#ifndef E92BA02D_8498_40BC_85B5_DAED223969ED
#define E92BA02D_8498_40BC_85B5_DAED223969ED

#include "../common/common.h"
#include "vector_base.h"


PE_BEGIN

template <typename T, uint32_t M, uint32_t N>
struct matrix_t {

  matrix_t() = default;

  PE_HOST_DEVICE matrix_t(T value) {
    PE_UNROLL
    for (uint32_t i = 0; i < N; i++) {
      PE_UNROLL
      for (uint32_t j = 0; j < M; j++) {
        rows[ i ][ j ] = (i == j) ? value : (T) 0;
      }
    }
  }



  union {
    vector_t<T, M> rows[ N ];
    T data[ M * N ];
  };

};

PE_END

#endif /* E92BA02D_8498_40BC_85B5_DAED223969ED */
