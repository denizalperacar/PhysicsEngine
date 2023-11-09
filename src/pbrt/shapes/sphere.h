#ifndef BE1B4E77_37A2_4D92_A1F3_9225DDDDE9CD
#define BE1B4E77_37A2_4D92_A1F3_9225DDDDE9CD

#include "pch.h"
#include "hittable.h"

PE_BEGIN

template <typename T, size_t A=sizeof(T)>
class sphere : public hittable<T, A> {

public:
  PE_HOST_DEVICE sphere() = default;
  PE_HOST_DEVICE sphere(const vector_t<T, 3, A>& center, T radius)
      : center(center), radius(radius)
  {}

  PE_HOST_DEVICE virtual bool hit(const ray_t<T>& r, T t_min, T t_max, hit_record<T, A>& rec) const override;

public:
  vector_t<T, 3, A> center;
  T radius;

};


template <typename T, size_t A>
PE_HOST_DEVICE bool sphere<T, A>::hit(const ray_t<T>& r, T t_min, T t_max, hit_record<T, A>& rec) const {
	vector_t<T, 3, A> oc = r.origin() - center;
	auto a = length_squared(r.direction());
	auto half_b = dot(oc, r.direction());
	auto c = length_squared(oc) - radius * radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false;
	auto sqrt_f = sqrtf(discriminant);
	// Find the nearest root that lies in the acceptable range.
	auto root = (-half_b - sqrt_f) / a;
	if (root < t_min || t_max < root) {
		root = (-half_b + sqrt_f) / a;
		if (root < t_min || t_max < root)
			return false;
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	vector_t<T, 3, A> outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);

	return true;
}

template <typename T, size_t A=sizeof(T)>
PE_HOST_DEVICE T hit_sphere(const vector_t<T, 3, A>& center, T radius, const ray_t<T>& r) {
  vector_t<T, 3, A> oc = r.origin() - center;
  auto a = length_squared(r.direction());
  auto half_b = dot(oc, r.direction());
  auto c = length_squared(oc) - radius * radius;
  auto discriminant = half_b * half_b - a * c;

  return (discriminant < 0) * (T)-1. + (discriminant >= 0) * (-half_b - sqrt(discriminant)) / a;
}

template <typename T, size_t A=sizeof(T)>
PE_HOST_DEVICE T hit_sphere_normal_coloring(const vector_t<T, 3, A>& center, T radius, const ray_t<T>& r) {
  vector_t<T, 3, A> oc = r.origin() - center;
  auto a = dot(r.direction(), r.direction());
  auto b = 2.0 * dot(oc, r.direction());
  auto c = dot(oc, oc) - radius * radius;
  auto discriminant = b * b - 4 * a * c;
  return (discriminant < 0) * (T)-1. + (discriminant >= 0) * (-b - sqrt(discriminant)) / ((T)2.0 * a);
}

PE_END

#endif /* BE1B4E77_37A2_4D92_A1F3_9225DDDDE9CD */
