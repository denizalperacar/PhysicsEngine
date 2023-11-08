#ifndef C61C8CB2_0C74_441C_8759_C05BA7E43E8B
#define C61C8CB2_0C74_441C_8759_C05BA7E43E8B


#include "pch.h"

PE_BEGIN

template <typename T, size_t ALIGNMENT=sizeof(T)>
struct hit_record {
  vector_t<T, 3, ALIGNMENT> p;
  vector_t<T, 3, ALIGNMENT> normal;
  T t;
  bool front_face;

  PE_HOST_DEVICE inline void set_face_normal(const ray_t<T>& r, const vector_t<T, 3>& outward_normal) {
    front_face = dot(r.direction, outward_normal) < 0;
    // normal = front_face ? outward_normal : -outward_normal;
    normal = front_face * outward_normal + (!front_face) * (-outward_normal);
  }
};

template <typename T, size_t ALIGNMENT=sizeof(T)>
class hittable {
public:
  PE_HOST_DEVICE virtual bool hit(const ray_t<T>& r, T t_min, T t_max, hit_record<T, ALIGNMENT>& rec) const = 0;
};

PE_END

#endif /* C61C8CB2_0C74_441C_8759_C05BA7E43E8B */
