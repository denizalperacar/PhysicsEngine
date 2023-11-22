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
  PE_HOST_DEVICE static constexpr uint32_t alignment() { return ALIGNMENT; } \


template <template <typename> class Child, typename T, uint32_t DIM, size_t ALIGNMENT=sizeof(T)>
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

template <template <typename> class Child, typename T, size_t ALIGNMENT>
struct alignas(ALIGNMENT) tuple_t<Child, T, 1, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 1;
  PE_TUPLE_BASE  
  union {T x, r, t; };
};


template <template <typename> class Child, typename T, size_t ALIGNMENT>
class alignas(ALIGNMENT) tuple_t<Child, T, 2, ALIGNMENT> {

public:
  static constexpr uint32_t DIM = 2;
  PE_TUPLE_BASE  
  union {T x, r; };
  union {T y, g; };
  PE_HOST_DEVICE tuple_t(T a, T b) : x(a), y(b) {}
};


template <template <typename> class Child, typename T, size_t ALIGNMENT>
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




PE_END


#endif /* EA97412A_D7FE_439C_9C70_D125C0801118 */
