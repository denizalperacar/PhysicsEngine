#ifndef AF1F8277_1C55_4D40_BD8A_1BD3DED792CE
#define AF1F8277_1C55_4D40_BD8A_1BD3DED792CE

#include "pch.h"

PE_BEGIN



template <typename T, uint32_t DIM, uint32_t ALIGNMENT=sizeof(T)>
class VectorBase {

  using value_type = T;
  using dimension = DIM;
  using alignment = ALIGNMENT;

  vector_t() = default;
  
  PE_HOST_DEVICE vector_t( T value ) {
    PE_UNROLL
    for (uint32_t i = 0; i < DIM; ++i) {
      (*this)[i] = value;
    }
  }

  PE_HOST_DEVICE vector_t( T* values ) {
    PE_UNROLL
    for (uint32_t i = 0; i < DIM; ++i) {
      (*this)[i] = values[i];
    }
  }

  template <typename U, uint32_t S, uint32_t A>
  PE_HOST_DEVICE vector_t(const vector_t<U,S,A> &other) {
    PE_UNROLL
    for (uint32_t i = 0; i < DIM; ++i) {
      (*this)[i] = i < M ? (T) other[i] : (T)0;
    }
  }

  PE_HOST_DEVICE static constexpr vector_t<T, N> zeros() {
    return vector_t<T, N>(0);
  }

  PE_HOST_DEVICE static constexpr vector_t<T, N> ones() {
    return vector_t<T, N>(1);
  }

  PE_HOST_DEVICE T& operator[](uint32_t index) {
    return ((T*) this)[index];
  }

  PE_HOST_DEVICE const T& operator[](uint32_t index) {
    return ((T*) this)[index];
  }

  PE_HOST_DEVICE T& operator()(uint32_t index) {
    return ((T*) this)[index];
  }

  PE_HOST_DEVICE const T& operator()(uint32_t index) {
    return ((T*) this)[index];
  }

  PE_HOST_DEVICE T* data() {
    return (T*) this;
  }

  PE_HOST_DEVICE const T* data() const {
    return (const T*) this;
  }

  PE_HOST_DEVICE T* begin() {
    return (T*) this;
  }

  PE_HOST_DEVICE const T* begin() const {
    return (const T*) this;
  } 

  PE_HOST_DEVICE T* end() {
    return (T*) this + DIM;
  }

  PE_HOST_DEVICE const T* end() const {
    return (const T*) this + DIM;
  }

  template <uint32_t start, uint32_t steps>
  PE_HOST_DEVICE vector_t<T, steps>& slice() {
    static_assert(start + steps <= DIM, "Slice out of bounds");
    return *(vector_t<T, steps>*)(data() + start);
  }

  template <uint32_t start, uint32_t steps>
  PE_HOST_DEVICE const vector_t<T, steps>& slice() const {
    static_assert(start + steps <= DIM, "Slice out of bounds");
    return *(vector_t<T, M>*)(data() + start);
  }


};


PE_END

#endif /* AF1F8277_1C55_4D40_BD8A_1BD3DED792CE */
