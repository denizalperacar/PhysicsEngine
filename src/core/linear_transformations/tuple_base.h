#ifndef EA97412A_D7FE_439C_9C70_D125C0801118
#define EA97412A_D7FE_439C_9C70_D125C0801118


#include "common.h"

PE_BEGIN

template <uint32_t N, typename T, typename... Ts>
using enable_if_size_and_type_match_t = std::enable_if_t<sizeof...(Ts) == N && conjunction<std::is_same<Ts, T>...>::value>;

#define PE_TUPLE_BASE \
  using value_type = T; \
  \
  tuple_t() = default; \
  \
  PE_HOST_DEVICE tuple_t( T value ) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = value;\
    }\
  }\
  \
  PE_HOST_DEVICE tuple_t( T* values ) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = values[i];\
    }\
  }\
  \
  template <typename U, uint32_t S, uint32_t A>\
  PE_HOST_DEVICE tuple_t(const tuple_t<Child, U, S, A> &other) {\
    PE_UNROLL\
    for (uint32_t i = 0; i < DIM; ++i) {\
      (*this)[i] = i < S ? (T) other[i] : (T)0;\
    }\
  }\
  \
  PE_HOST_DEVICE static constexpr tuple_t<Child, T, DIM> zeros() {\
    return tuple_t<Child, T, DIM>(0);\
  }\
  \
  PE_HOST_DEVICE static constexpr tuple_t<Child, T, DIM> ones() {\
    return tuple_t<Child, T, DIM>(1);\
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
  PE_HOST_DEVICE static constexpr uint32_t size() { return DIM; } \
  PE_HOST_DEVICE static constexpr uint32_t alignment() { return ALIGNMENT; }


template <template <class, uint32_t, size_t> class Child, typename T, uint32_t DIM, size_t ALIGNMENT=sizeof(T)>
class alignas(ALIGNMENT) tuple_t {

public:
  PE_TUPLE_BASE
  T elements[DIM];
  template <typename...Tn, typename = enable_if_size_and_type_match_t<DIM, T, Tn...>>
  PE_HOST_DEVICE tuple_t(Tn... values) : elements{values...} {}
  PE_HOST_DEVICE tuple_t(std::initializer_list<T> values) {
    static_assert(values.size() == DIM, "Initializer list size does not match vector size");
    std::copy(values.begin(), values.end(), elements);
  }
};

template <template <class, uint32_t, size_t> class Child, typename T, size_t ALIGNMENT>
struct alignas(ALIGNMENT) tuple_t<Child, T, 1, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 1;
  PE_TUPLE_BASE  
  union {T x, r, t; };
};


template <template <class, uint32_t, size_t> class Child, typename T, size_t ALIGNMENT>
class alignas(ALIGNMENT) tuple_t<Child, T, 2, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 2;
  PE_TUPLE_BASE  
  union {T x, r; };
  union {T y, g; };
  PE_HOST_DEVICE tuple_t(T a, T b) : x(a), y(b) {}
};


template <template <class, uint32_t, size_t> class Child, typename T, size_t ALIGNMENT>
class alignas(ALIGNMENT) tuple_t<Child, T, 3, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 3;
  PE_TUPLE_BASE  
  union {T x, r; };
  union {T y, g; };
  union {T z, b; };
  PE_HOST_DEVICE tuple_t(T a, T b, T c) : x(a), y(b), z(c) {}
  template<size_t A> PE_HOST_DEVICE tuple_t(tuple_t<Child, T, 2, A> a, T b) : x((T)a.x), y((T)a.y), z(b) {}
  template<size_t A> PE_HOST_DEVICE tuple_t(T a, tuple_t<Child, T, 2, A> b) : x(a), y((T)b.x), z((T)b.y) {}

};

template <template <class, uint32_t, size_t> class Child, typename T, size_t ALIGNMENT>
struct alignas(ALIGNMENT) tuple_t<Child, T, 4, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 4;
  PE_TUPLE_BASE  
  union {T x, r; };
  union {T y, g; };
  union {T z, b; };
  union {T w, a; };

  PE_HOST_DEVICE tuple_t(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
  template<size_t A> PE_HOST_DEVICE tuple_t(tuple_t<Child, T, 2, A> a, T b, T c) : x(a.x), y(a.y), z(b), w(c) {}
  template<size_t A> PE_HOST_DEVICE tuple_t(T a, tuple_t<Child, T, 2, A> b, T c) : x(a), y(b.x), z(b.y), w(c) {}
  template<size_t A> PE_HOST_DEVICE tuple_t(T a, T b, tuple_t<Child, T, 2, A> c) : x(a), y(b), z(c.x), w(c.y) {}
  template<size_t A> PE_HOST_DEVICE tuple_t(tuple_t<Child, T, 3, A> a, T b) : x(a.x), y(a.y), z(a.z), w(b) {}
  template<size_t A> PE_HOST_DEVICE tuple_t(T a, tuple_t<Child, T, 3, A> b) : x(a), y(b.x), z(b.y), w(b.z) {}
  template<size_t A, size_t B> PE_HOST_DEVICE tuple_t(tuple_t<Child, T, 2, A> a, tuple_t<Child, T, 2, B> b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
  template<size_t A> PE_HOST_DEVICE tuple_t(tuple_t<Child, T, 3, A>& a) : x(a.x), y(a.y), z(a.z), w((T)0) {}
  template<size_t A> PE_HOST_DEVICE tuple_t<Child, T, 4, ALIGNMENT>& operator=(const tuple_t<Child, T, 3, A>& a) {
    x = a.x;
    y = a.y;
    z = a.z;
    w = (T)0;
    return *this;
  }
};


// define the elemtwise operations on vector
#define TTUP tuple_t<Child, T, D, A>
#define CTUP Child<decltype(T{} + U{}), D, A>
#define BTUP tuple_t<Child, bool, D, A>

#define ELEMENTWISE_OP(operator, result_type, expression, ...) \
template< template <class, uint32_t, size_t> class Child, typename U, typename T, uint32_t D, size_t A> \
PE_HOST_DEVICE auto operator(__VA_ARGS__) -> result_type { \
  result_type result; \
  PE_UNROLL \
  for (uint32_t ind = 0; ind < D; ind++) { \
    result[ind] = expression; \
  } \
  return result;\
}

ELEMENTWISE_OP(operator+, TTUP, a[ind] + b[ind], const TTUP &a, const TTUP &b)
ELEMENTWISE_OP(operator+, CTUP, a[ind] + b[ind], const TTUP &a, const CTUP &b)
ELEMENTWISE_OP(operator+, CTUP, a[ind] + b[ind], const CTUP &a, const TTUP &b)

ELEMENTWISE_OP(operator+, CTUP, a + b[ind], T a, const CTUP &b)
ELEMENTWISE_OP(operator+, CTUP, a[ind] + b, const CTUP &a, T b)
ELEMENTWISE_OP(operator+, TTUP, a + b[ind], T a, const TTUP &b)
ELEMENTWISE_OP(operator+, TTUP, a[ind] + b, const TTUP &a, T b)

// TODO : Add all the permuataions of cases below for CTUP and TTUP

ELEMENTWISE_OP(operator-, TTUP, a[ind] - b[ind], const TTUP &a, const TTUP &b)
ELEMENTWISE_OP(operator-, TTUP, a - b[ind], T a, const TTUP &b)
ELEMENTWISE_OP(operator-, TTUP, a[ind] - b, const TTUP &a, T b)
ELEMENTWISE_OP(operator-, TTUP, -a[ind], const TTUP &a)

ELEMENTWISE_OP(operator*, TTUP, a[ind] * b[ind], const TTUP &a, const TTUP &b)
ELEMENTWISE_OP(operator*, TTUP, a * b[ind], T a, const TTUP &b)
ELEMENTWISE_OP(operator*, TTUP, a[ind] * b, const TTUP &a, T b)

ELEMENTWISE_OP(operator/, TTUP, a[ind] / b[ind], const TTUP &a, const TTUP &b)
ELEMENTWISE_OP(operator/, TTUP, a / b[ind], T a, const TTUP &b)
ELEMENTWISE_OP(operator/, TTUP, a[ind] / b, const TTUP &a, T b)

ELEMENTWISE_OP(min, TTUP, min(a[ind], b[ind]), const TTUP& a, const TTUP& b)
ELEMENTWISE_OP(min, TTUP, min(a[ind], b), const TTUP& a, T b)
ELEMENTWISE_OP(min, TTUP, min(a, b[ind]), T a, const TTUP& b)

ELEMENTWISE_OP(atan2, TTUP, atan2(a[ind], b[ind]), const TTUP& a, const TTUP& b)
ELEMENTWISE_OP(atan2, TTUP, atan2(a[ind], b), const TTUP& a, T b)
ELEMENTWISE_OP(atan2, TTUP, atan2(a, b[ind]), T a, const TTUP& b)

ELEMENTWISE_OP(max, TTUP, max(a[ind], b[ind]), const TTUP& a, const TTUP& b)
ELEMENTWISE_OP(max, TTUP, max(a[ind], b), const TTUP& a, T b)
ELEMENTWISE_OP(max, TTUP, max(a, b[ind]), T a, const TTUP& b)

ELEMENTWISE_OP(pow, TTUP, pow(a[ind], b[ind]), const TTUP& a, const TTUP& b)
ELEMENTWISE_OP(pow, TTUP, pow(a[ind], b), const TTUP& a, T b)
ELEMENTWISE_OP(pow, TTUP, pow(a, b[ind]), T a, const TTUP& b)

ELEMENTWISE_OP(distance, TTUP, distance(a[ind], b[ind]), const TTUP& a, const TTUP& b)
ELEMENTWISE_OP(distance, TTUP, distance(a[ind], b), const TTUP& a, T b)
ELEMENTWISE_OP(distance, TTUP, distance(a, b[ind]), T a, const TTUP& b)

ELEMENTWISE_OP(copysign, TTUP, copysign(a[ind], b[ind]), const TTUP& a, const TTUP& b)
ELEMENTWISE_OP(copysign, TTUP, copysign(a[ind], b), const TTUP& a, T b)
ELEMENTWISE_OP(copysign, TTUP, copysign(a, b[ind]), T a, const TTUP& b)

ELEMENTWISE_OP(sign, TTUP, sign(a[ind]), const TTUP& a)
ELEMENTWISE_OP(floor, TTUP, floor(a[ind]), const TTUP& a)
ELEMENTWISE_OP(ceil, TTUP, ceil(a[ind]), const TTUP& a)
ELEMENTWISE_OP(abs, TTUP, abs(a[ind]), const TTUP& a)
ELEMENTWISE_OP(sin, TTUP, sin(a[ind]), const TTUP& a)
ELEMENTWISE_OP(asin, TTUP, asin(a[ind]), const TTUP& a)
ELEMENTWISE_OP(cos, TTUP, cos(a[ind]), const TTUP& a)
ELEMENTWISE_OP(acos, TTUP, acos(a[ind]), const TTUP& a)
ELEMENTWISE_OP(tan, TTUP, tan(a[ind]), const TTUP& a)
ELEMENTWISE_OP(tanh, TTUP, tanh(a[ind]), const TTUP& a)
ELEMENTWISE_OP(atan, TTUP, atan(a[ind]), const TTUP& a)
ELEMENTWISE_OP(sqrt, TTUP, sqrt(a[ind]), const TTUP& a)
ELEMENTWISE_OP(exp, TTUP, exp(a[ind]), const TTUP& a)
ELEMENTWISE_OP(log, TTUP, log(a[ind]), const TTUP& a)
ELEMENTWISE_OP(log2, TTUP, log2(a[ind]), const TTUP& a)
ELEMENTWISE_OP(log10, TTUP, log10(a[ind]), const TTUP& a)
ELEMENTWISE_OP(isfinite, BTUP, isfinite(a[ind]), const TTUP& a)

ELEMENTWISE_OP(clamp, TTUP, clamp(a[ind], b[ind], c[ind]), const TTUP& a, const TTUP& b, const TTUP& c)
ELEMENTWISE_OP(clamp, TTUP, clamp(a[ind], b[ind], c), const TTUP& a, const TTUP& b, T c)
ELEMENTWISE_OP(clamp, TTUP, clamp(a[ind], b, c[ind]), const TTUP& a, T b, const TTUP& c)
ELEMENTWISE_OP(clamp, TTUP, clamp(a[ind], b, c), const TTUP& a, T b, T c)

ELEMENTWISE_OP(mix, TTUP, a[ind] * ((T)1 - c[ind]) + b[ind] * c[ind], const TTUP& a, const TTUP& b, const TTUP& c)
ELEMENTWISE_OP(mix, TTUP, a[ind] * ((T)1 - c) + b[ind] * c, const TTUP& a, const TTUP& b, T c)

ELEMENTWISE_OP(fma, TTUP, fma(a[ind], b[ind], c[ind]), const TTUP& a, const TTUP& b, const TTUP& c)
ELEMENTWISE_OP(fma, TTUP, fma(a[ind], b[ind], c), const TTUP& a, const TTUP& b, T c)
ELEMENTWISE_OP(fma, TTUP, fma(a[ind], b, c[ind]), const TTUP& a, T b, const TTUP& c)
ELEMENTWISE_OP(fma, TTUP, fma(a[ind], b, c), const TTUP& a, T b, T c)
ELEMENTWISE_OP(fma, TTUP, fma(a, b[ind], c[ind]), T a, const TTUP& b, const TTUP& c)
ELEMENTWISE_OP(fma, TTUP, fma(a, b[ind], c), T a, const TTUP& b, T c)
ELEMENTWISE_OP(fma, TTUP, fma(a, b, c[ind]), T a, T b, const TTUP& c)




#undef TTUP
#undef BTUP
#undef PE_TUPLE_BASE
PE_END


#endif /* EA97412A_D7FE_439C_9C70_D125C0801118 */
