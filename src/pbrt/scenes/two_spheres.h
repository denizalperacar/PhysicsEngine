#ifndef F64C426A_7A84_4FE0_A188_5668A2274EE1
#define F64C426A_7A84_4FE0_A188_5668A2274EE1

#include "pch.h"
#include "../shapes/sphere.h"
#include "../shapes/hittable.h"

PE_BEGIN

template <typename T>
PE_KERNEL void two_spheres(hittable<T>** list, hittable<T>** world) {
	
	uint32_t i = 0;
	list[i++] = new sphere<T>(vector_t<T, 3>(0.f, 0.f, -1.0f), 0.5f);
	list[i++] = new sphere<T>(vector_t<T, 3>(0.f, -100.5f, -1.f), 100.f);
	*world = new hittable_list(list, i);
}

PE_END

#endif /* F64C426A_7A84_4FE0_A188_5668A2274EE1 */
