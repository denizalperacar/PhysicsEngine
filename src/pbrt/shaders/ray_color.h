#ifndef A068C48D_C2FE_43CA_9A9D_2DCA1ECE7701
#define A068C48D_C2FE_43CA_9A9D_2DCA1ECE7701

#include "pch.h"
#include "../shapes/hittable.h"

PE_BEGIN

template <typename T>
PE_HOST_DEVICE vector_t<T, 3> ray_color(const ray_t<T>& r, hittable<T>* world) {

  hit_record<T> rec;

  if (world->hit(r, 0, infinity, rec)) {
    return (T)0.5f * (rec.normal + vector_t<T, 3>((T)1));
  }

  vector_t<T, 3> unit_direction = normalize(r.direction());
  T t = (T)0.5 * (unit_direction.y + (T)1.0);
  return ((T)1.0 - t) * vector_t<T, 3>((T)1.0) + t * vector_t<T, 3>((T)0.5, (T)0.7, (T)1.0);

}

PE_END

#endif /* A068C48D_C2FE_43CA_9A9D_2DCA1ECE7701 */
