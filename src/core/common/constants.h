#ifndef AD73A52F_B9EF_43C8_A4E1_E903BB5D4FEB
#define AD73A52F_B9EF_43C8_A4E1_E903BB5D4FEB


#include "namespaces.h"
#include "cstdint"

PE_BEGIN

// constants
constexpr double pi = 3.14159265358979323846;
constexpr float infinity = std::numeric_limits<float>::infinity();

// Thread constants
constexpr uint32_t NUM_THREADS_MIN = 32;
constexpr uint32_t NUM_THREADS_64  = 64;
constexpr uint32_t NUM_THREADS_128 = 128;
constexpr uint32_t NUM_THREADS_256 = 256;
constexpr uint32_t NUM_THREADS_512 = 512;
constexpr uint32_t NUM_THREADS_1024 = 1024;
constexpr uint32_t NUM_THREADS_MAX = 1024;


PE_END


#endif /* AD73A52F_B9EF_43C8_A4E1_E903BB5D4FEB */
