#ifndef B0AF69A6_577B_4BF9_AF88_BB09112AEE6E
#define B0AF69A6_577B_4BF9_AF88_BB09112AEE6E

#include "../common/common.h"
#include "rotation_dyads.h"

PE_BEGIN

template <typename T>
struct htm_t {
  
  htm_t() = default;
  


  matrix_t<T, 4, 4> matrix= matrix_t<T, 4, 4>::identity();
};

PE_END

#endif /* B0AF69A6_577B_4BF9_AF88_BB09112AEE6E */
