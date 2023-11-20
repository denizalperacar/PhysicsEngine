#ifndef D67A76C9_4D71_4084_813D_2E63C21A2759
#define D67A76C9_4D71_4084_813D_2E63C21A2759

#include "pch.h"
#include "../shaders/color_render.h"
#include "../shapes/hittable_list.h"
#include "../shaders/get_color.h"
#include "../shaders/ray_color.h"
#include "../scenes/two_spheres.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

PE_BEGIN

template <typename T=float>
PE_KERNEL void renderer(uchar4* results, hittable<T>** world) {
  uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t j = threadIdx.y + blockDim.y * blockIdx.y;
  uint32_t idx = j * gridDim.x * blockDim.x + i;

  if (i < DEFAULT_IMAGE_WIDTH && j < DEFAULT_IMAGE_HEIGHT) {
    // Camera

    auto viewport_height = 2.0f;
    auto viewport_width = DEFAULT_IMAGE_ASPECT_RATIO * viewport_height;
    auto focal_length = 1.0f;

    auto origin = vector_t<T, 3>(0.f, 0.f, 0.f);
    auto horizontal = vector_t<T, 3>(viewport_width, 0.f, 0.f);
    auto vertical = vector_t<T, 3>(0, viewport_height, 0.f);
    auto lower_left_corner = origin - horizontal / (T)2.f - vertical / (T)2.f - vector_t<T, 3>((T)0.f, (T)0.f, focal_length);

    render_color c;
    auto u = ((T)(i) / (T)(DEFAULT_IMAGE_WIDTH - 1));
    auto v = ((T)(DEFAULT_IMAGE_HEIGHT - j) / (T)(DEFAULT_IMAGE_HEIGHT - 1));
    ray_t<T> r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    vector_t<T, 3> pixel_color = ray_color<T>(r, *world);
    results[idx] = get_color<T>(pixel_color, 1);
	}
}


PE_END

#endif /* D67A76C9_4D71_4084_813D_2E63C21A2759 */
