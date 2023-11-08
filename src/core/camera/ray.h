#ifndef B1AB7642_6C36_4D30_B8E7_5911833C7AA4
#define B1AB7642_6C36_4D30_B8E7_5911833C7AA4

#include "common.h"
#include "vector_base.h"

PE_BEGIN

template <typename T>
class ray_t {

public:
	PE_HOST_DEVICE ray_t<T>() = default;
	PE_HOST_DEVICE ray_t<T>(const vector_t<T, 3>& ray_origin, const vector_t<T, 3>& ray_direction, T time = static_cast<T>(0.))
		: orig(ray_origin), dir(ray_direction)
	{}

	PE_HOST_DEVICE vector_t<T, 3> origin() const { return orig; }
	PE_HOST_DEVICE vector_t<T, 3> direction() const { return dir; }
	PE_HOST_DEVICE T time() const { return tm; }
	PE_HOST_DEVICE vector_t<T, 3> at(T t) const {
		return orig + t * dir;
	}

public:
	vector_t<T, 3> orig;
	vector_t<T, 3> dir;
	T tm;

};

PE_END


#endif /* B1AB7642_6C36_4D30_B8E7_5911833C7AA4 */
