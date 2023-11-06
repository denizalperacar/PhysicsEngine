#ifndef F94345FF_187E_4988_B654_1C2243B316B1
#define F94345FF_187E_4988_B654_1C2243B316B1

#include "common.h"
#include "../linear_transformations/vector_base.h"
#include "../linear_transformations/matrix_base.h"
#include "../linear_transformations/rotation_dyads.h"
#include "../linear_transformations/quaternion.h"
#include "../linear_transformations/homogeneous_transformation_matrix.h"
#include "../coordinate_frame/global_frame.h"

// define library specific types 

PE_BEGIN

template <typename T, size_t A=sizeof(T)>
using color_t = vector_t<T, 4, A>; 



PE_END

#endif /* F94345FF_187E_4988_B654_1C2243B316B1 */
