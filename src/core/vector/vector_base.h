#ifndef AF1F8277_1C55_4D40_BD8A_1BD3DED792CE
#define AF1F8277_1C55_4D40_BD8A_1BD3DED792CE

#include "common.h"

PE_BEGIN



#define PE_VECTOR_BASE \
  using value_type = T; \
  \
  vector_t() = default; \
  \
  PE_HOST_DEVICE vector_t( T value ) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = value;\
    }\
  }\
  \
  PE_HOST_DEVICE vector_t( T* values ) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = values[i];\
    }\
  }\
  \
  template <typename U, uint32_t S, uint32_t A>\
  PE_HOST_DEVICE vector_t(const vector_t<U,S,A> &other) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = i < S ? (T) other[i] : (T)0;\
    }\
  }\
  \
  PE_HOST_DEVICE static constexpr vector_t<T, DIM> zeros() {\
    return vector_t<T, DIM>(0);\
  }\
  \
  PE_HOST_DEVICE static constexpr vector_t<T, DIM> ones() {\
    return vector_t<T, DIM>(1);\
  }\
  \
  PE_HOST_DEVICE T& operator[](uint32_t index) {\
    return ((T*) this)[index];\
  }\
  \
  PE_HOST_DEVICE const T& operator[](uint32_t index) const {\
    return ((T*) this)[index];\
  }\
  \
  PE_HOST_DEVICE T& operator()(uint32_t index) {\
    return ((T*) this)[index];\
  }\
  \
  PE_HOST_DEVICE const T& operator()(uint32_t index) const {\
    return ((T*) this)[index];\
  }\
  \
  PE_HOST_DEVICE T* data() {\
    return (T*) this;\
  }\
  \
  PE_HOST_DEVICE const T* data() const {\
    return (const T*) this;\
  }\
  \
  PE_HOST_DEVICE T* begin() {\
    return (T*) this;\
  }\
  \
  PE_HOST_DEVICE const T* begin() const {\
    return (const T*) this;\
  } \
  \
  PE_HOST_DEVICE T* end() {\
    return (T*) this + DIM;\
  }\
  \
  PE_HOST_DEVICE const T* end() const {\
    return (const T*) this + DIM;\
  }\
  \
  template <uint32_t start, uint32_t steps> \
  PE_HOST_DEVICE vector_t<T, steps>& slice() { \
    static_assert(start + steps <= DIM, "Slice out of bounds"); \
    return *(vector_t<T, steps>*)(data() + start); \
  } \
  \
  template <uint32_t start, uint32_t steps> \
  PE_HOST_DEVICE const vector_t<T, steps>& slice() const { \
    static_assert(start + steps <= DIM, "Slice out of bounds"); \
    return *(vector_t<T, steps>*)(data() + start); \
  } \
  PE_HOST_DEVICE static constexpr uint32_t size() { return DIM; } \
  PE_HOST_DEVICE static constexpr uint32_t alignment() { return ALIGNMENT; } \



template <typename T, uint32_t DIM, uint32_t ALIGNMENT=sizeof(T)>
struct alignas(ALIGNMENT) vector_t {

public:
  PE_VECTOR_BASE

private:  
  T elements[DIM];
};

template <typename T, uint32_t ALIGNMENT>
struct alignas(ALIGNMENT) vector_t<T, 1, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 1;
  PE_VECTOR_BASE  
  union {T x, r, t; };
};


template <typename T, uint32_t ALIGNMENT>
struct alignas(ALIGNMENT) vector_t<T, 2, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 2;
  PE_VECTOR_BASE  
  union {T x, r; };
  union {T y, g; };
  PE_HOST_DEVICE vector_t(T a, T b) : x(a), y(b) {}
};

template <typename T, uint32_t ALIGNMENT>
struct alignas(ALIGNMENT) vector_t<T, 3, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 3;
  PE_VECTOR_BASE  
  union {T x, r; };
  union {T y, g; };
  union {T z, b; };
  PE_HOST_DEVICE vector_t(T a, T b, T c) : x(a), y(b), z(c) {}
  template<size_t A> PE_HOST_DEVICE vector_t(vector_t<T, 2, A> a, T b) : x(a.x), y(a.y), z(b) {}
  template<size_t A> PE_HOST_DEVICE vector_t(T a, vector_t<T, 2, A> b) : x(a), y(b.x), z(b.y) {}

};


template <typename T, uint32_t ALIGNMENT>
struct alignas(ALIGNMENT) vector_t<T, 4, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 4;
  PE_VECTOR_BASE  
  union {T x, r; };
  union {T y, g; };
  union {T z, b; };
  union {T w, a; };

  PE_HOST_DEVICE vector_t(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
  template<size_t A> PE_HOST_DEVICE vector_t(vector_t<T, 2, A> a, T b, T c) : x(a.x), y(a.y), z(b), w(c) {}
  template<size_t A> PE_HOST_DEVICE vector_t(T a, vector_t<T, 2, A> b, T c) : x(a), y(b.x), z(b.y), w(c) {}
  template<size_t A> PE_HOST_DEVICE vector_t(T a, T b, vector_t<T, 2, A> c) : x(a), y(b), z(c.x), w(c.y) {}
  template<size_t A> PE_HOST_DEVICE vector_t(vector_t<T, 3, A> a, T b) : x(a.x), y(a.y), z(a.z), w(b) {}
  template<size_t A> PE_HOST_DEVICE vector_t(T a, vector_t<T, 3, A> b) : x(a), y(b.x), z(b.y), w(b.z) {}
  template<size_t A, size_t B> PE_HOST_DEVICE vector_t(vector_t<T, 2, A> a, vector_t<T, 2, B> b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
};

#undef PE_VECTOR_BASE


template <typename T> PE_HOST_DEVICE T min(T a, T b) { return std::min(a, b); }
template <typename T> PE_HOST_DEVICE T max(T a, T b) { return std::max(a, b); }
template <typename T> PE_HOST_DEVICE T pow(T a, T b) { return std::pow(a, b); }
template <typename T> PE_HOST_DEVICE T distance(T a, T b) { return std::abs(a - b); }
template <typename T> PE_HOST_DEVICE T copysign(T a, T b) { return std::copysign(a, b); }
template <typename T> PE_HOST_DEVICE T sign(T a) { return std::copysign((T)1, a); }
template <typename T> PE_HOST_DEVICE T floor(T a) { return std::floor(a); }
template <typename T> PE_HOST_DEVICE T ceil(T a) { return std::ceil(a); }
template <typename T> PE_HOST_DEVICE T abs(T a) { return std::abs(a); }
template <typename T> PE_HOST_DEVICE T sin(T a) { return std::sin(a); }
template <typename T> PE_HOST_DEVICE T asin(T a) { return std::asin(a); }
template <typename T> PE_HOST_DEVICE T cos(T a) { return std::cos(a); }
template <typename T> PE_HOST_DEVICE T acos(T a) { return std::acos(a); }
template <typename T> PE_HOST_DEVICE T tan(T a) { return std::tan(a); }
template <typename T> PE_HOST_DEVICE T atan(T a) { return std::atan(a); }
template <typename T> PE_HOST_DEVICE T sqrt(T a) { return std::sqrt(a); }
template <typename T> PE_HOST_DEVICE T exp(T a) { return std::exp(a); }
template <typename T> PE_HOST_DEVICE T log(T a) { return std::log(a); }
template <typename T> PE_HOST_DEVICE T exp2(T a) { return std::exp2(a); }
template <typename T> PE_HOST_DEVICE T log2(T a) { return std::log2(a); }
template <typename T> PE_HOST_DEVICE T clamp(T a, T b, T c) { return a < b ? b : (c < a ? c : a); }
template <typename T> PE_HOST_DEVICE T mix(T a, T b, T c) { return a * ((T)1 - c) + b * c; }

template <typename T> PE_HOST_DEVICE T isfinite(T a) {
#if defined(__CUDA_ARCH__)
  return ::isfinite(a);
#else
  return std::isfinite(a);
#endif
}

inline PE_HOST_DEVICE float fma(float a, float b, float c) { return fmaf(a, b, c); }


// define the elemtwise operations on vector
#define TVEC vector_t<T, D, A>
#define BVEC vector_t<bool, D, A>

#define ELEMENTWISE_OP(operator, result_type, expression, ...) \
template<typename T, uint32_t D, size_t A> \
PE_HOST_DEVICE result_type operator(__VA_ARGS__) { \
  result_type result; \
  PE_UNROLL \
  for (uint32_t ind = 0; ind < D; ind++) { \
    result[ind] = expression; \
  } \
  return result;\
}

ELEMENTWISE_OP(operator+, TVEC, a[ind] + b[ind], const TVEC &a, const TVEC &b)
ELEMENTWISE_OP(operator+, TVEC, a + b[ind], T a, const TVEC &b)
ELEMENTWISE_OP(operator+, TVEC, a[ind] + b, const TVEC &a, T b)

ELEMENTWISE_OP(operator-, TVEC, a[ind] - b[ind], const TVEC &a, const TVEC &b)
ELEMENTWISE_OP(operator-, TVEC, a - b[ind], T a, const TVEC &b)
ELEMENTWISE_OP(operator-, TVEC, a[ind] - b, const TVEC &a, T b)
ELEMENTWISE_OP(operator-, TVEC, -a[ind], const TVEC &a)

ELEMENTWISE_OP(operator*, TVEC, a[ind] * b[ind], const TVEC &a, const TVEC &b)
ELEMENTWISE_OP(operator*, TVEC, a * b[ind], T a, const TVEC &b)
ELEMENTWISE_OP(operator*, TVEC, a[ind] * b, const TVEC &a, T b)

ELEMENTWISE_OP(operator/, TVEC, a[ind] / b[ind], const TVEC &a, const TVEC &b)
ELEMENTWISE_OP(operator/, TVEC, a / b[ind], T a, const TVEC &b)
ELEMENTWISE_OP(operator/, TVEC, a[ind] / b, const TVEC &a, T b)

ELEMENTWISE_OP(min, TVEC, min(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(min, TVEC, min(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(min, TVEC, min(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(max, TVEC, max(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(max, TVEC, max(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(max, TVEC, max(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(pow, TVEC, pow(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(pow, TVEC, pow(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(pow, TVEC, pow(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(distance, TVEC, distance(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(distance, TVEC, distance(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(distance, TVEC, distance(a, b[ind]), T a, const TVEC& b)

ELEMENTWISE_OP(copysign, TVEC, copysign(a[ind], b[ind]), const TVEC& a, const TVEC& b)
ELEMENTWISE_OP(copysign, TVEC, copysign(a[ind], b), const TVEC& a, T b)
ELEMENTWISE_OP(copysign, TVEC, copysign(a, b[ind]), T a, const TVEC& b)






ELEMENTWISE_OP(fma, TVEC, fma(a[ind], b[ind], c[ind]), const TVEC& a, const TVEC& b, const TVEC& c)
ELEMENTWISE_OP(fma, TVEC, fma(a[ind], b[ind], c), const TVEC& a, const TVEC& b, T c)
ELEMENTWISE_OP(fma, TVEC, fma(a[ind], b, c[ind]), const TVEC& a, T b, const TVEC& c)
ELEMENTWISE_OP(fma, TVEC, fma(a[ind], b, c), const TVEC& a, T b, T c)
ELEMENTWISE_OP(fma, TVEC, fma(a, b[ind], c[ind]), T a, const TVEC& b, const TVEC& c)
ELEMENTWISE_OP(fma, TVEC, fma(a, b[ind], c), T a, const TVEC& b, T c)
ELEMENTWISE_OP(fma, TVEC, fma(a, b, c[ind]), T a, T b, const TVEC& c)


PE_END

#endif /* AF1F8277_1C55_4D40_BD8A_1BD3DED792CE */
