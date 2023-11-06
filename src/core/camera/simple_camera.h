#ifndef E580B530_A848_44ED_9E06_A7AAEF3057CE
#define E580B530_A848_44ED_9E06_A7AAEF3057CE


#include "../common/pch.h"
#include "../coordinate_frame/absolute_frame.h"
#include "ray.h"


PE_BEGIN

template <typename T>
class SimpleCamera {

public: 
  PE_HOST_DEVICE SimpleCamera() = delete;
  SimpleCamera(
    vector_t<T, 3> lookfrom,
    vector_t<T, 3> lookat,
    vector_t<T, 3> up_vector,
    T vertical_field_of_view,
    T aspect_ratio,
    T aperture,
    T focus_distance
  ) {
    vector_t<T, 3> u_, v_, w_;
    T theta = degrees_to_radians(vertical_field_of_view);
		T h = tan(theta / (T)2.);
		T viewport_height{(T)2.0 * h};
		T viewport_width{ (T) aspect_ratio * viewport_height };
  	w_ = unit_vector(lookfrom - lookat);
		u_ = unit_vector(cross(up_vector, w_));
		v_ = unit_vector(cross(w_, u_));  
    origin = lookfrom;
    horizontal = focus_distance * viewport_width * u_;
    vertical = focus_distance * viewport_height * v_;
    lower_left_corner = origin - horizontal / (T)2. - vertical / (T)2. - focus_distance * w_;
    lens_radius = aperture / (T)2.;

  }

  ray_t<T> get_ray(T s, T t) const {

		vector_t<T, 3> rd = lens_radius;
		vector_t<T, 3> offset = u_ * rd.x() + v_ * rd.y();

		return ray_t<T>(
			origin + offset,
			lower_left_corner + s * horizontal + t * vertical - origin - offset, 
      (T)0
		);
	}

  htm_t<T> get_htm() const {
    htm_t<T> htm;
    htm.set_position(origin);
    htm.set_rotation(u_, v_, w_);
    return htm;
  }

private:
  vector_t<T, 3> origin, u_, v_, w_;
  vector_t<T, 3> horizontal;
  vector_t<T, 3> vertical;
  vector_t<T, 3> lower_left_corner;
  T lens_radius;

};

PE_END


#endif /* E580B530_A848_44ED_9E06_A7AAEF3057CE */
