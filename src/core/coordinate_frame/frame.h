#ifndef C42C2054_3026_478C_82CD_E6BB0A8DD2DE
#define C42C2054_3026_478C_82CD_E6BB0A8DD2DE

/*
@brief: Implementation of reference frames (static and moving)
@file: frame.h
@author: Deniz A. ACAR
*/


#include <../common/pch.h>


PE_BEGIN

template <typename T, uint32_t ALIGNMENT>
struct Frame {
  using value_type = T;
  
  


  vector_t<T, 4, ALIGNMENT> position;
  // define quaternons for here

};

PE_END


#endif /* C42C2054_3026_478C_82CD_E6BB0A8DD2DE */


