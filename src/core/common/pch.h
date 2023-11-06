#ifndef F94345FF_187E_4988_B654_1C2243B316B1
#define F94345FF_187E_4988_B654_1C2243B316B1

#include "common.h"
#include "../linear_transformations/vector_base.h"
#include "../linear_transformations/matrix_base.h"
#include "../linear_transformations/rotation_dyads.h"
#include "../linear_transformations/quaternion.h"
#include "../linear_transformations/homogeneous_transformation_matrix.h"
#include "../coordinate_frame/global_frame.h"

// define project specific types 
PE_BEGIN

template <typename T, size_t A=sizeof(T)>
using color_t = vector_t<T, 4, A>; 

#define color(name, r, g, b, a) \
template <typename T, size_t A> \
constexpr color_t<T, A> name = color_t<T, A>{r, g, b, a};

color(BLACK, 0, 0, 0, 1)
color(WHITE, 1, 1, 1, 1)
color(RED, 1, 0, 0, 1)
color(GREEN, 0, 1, 0, 1)
color(BLUE, 0, 0, 1, 1)
color(YELLOW, 1, 1, 0, 1)
color(CYAN, 0, 1, 1, 1)
color(MAGNETA, 1, 0, 1, 1)
color(GRAY, 0.5, 0.5, 0.5, 1)
color(DARK_GRAY, 0.25, 0.25, 0.25, 1)
color(LIGHT_GRAY, 0.75, 0.75, 0.75, 1)

#undef color

PE_END

#endif /* F94345FF_187E_4988_B654_1C2243B316B1 */
