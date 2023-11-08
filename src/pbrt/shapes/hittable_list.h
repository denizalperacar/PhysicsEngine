#ifndef DB914FD3_96F8_4AB0_A66A_A184264CFC75
#define DB914FD3_96F8_4AB0_A66A_A184264CFC75

#include "pch.h"
#include "hittable.h"

PE_BEGIN

template <typename T, size_t A=sizeof(T)>
class hittable_list : public hittable<T, A> {
public:
  hittable_list() {}
  hittable_list(hittable<T, A>** l, size_t n) : list(l), list_size(n) {}

  PE_HOST_DEVICE virtual bool hit(const ray_t<T>& r, T t_min, T t_max, hit_record<T, A>& rec) const override;

  memory_t<hittable<T, A>*> list;
  size_t list_size;
};

template <typename T, size_t A>
PE_HOST_DEVICE bool hittable_list<T, A>::hit(const ray_t<T>& r, T t_min, T t_max, hit_record<T, A>& rec) const {
  hit_record<T, A> temp_rec;
  bool hit_anything = false;
  T closest_so_far = t_max;
  for (size_t i = 0; i < list_size; ++i) {
    if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }
  return hit_anything;
}

PE_END

#endif /* DB914FD3_96F8_4AB0_A66A_A184264CFC75 */
