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
PE_KERNEL void renderer(cudaSurfaceObject_t surface_object, hittable<T>** world) {
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
      surf2Dwrite(get_color<T>(pixel_color, 1), surface_object, i * sizeof(uchar4), j);
	}
}


template <typename T>
memory_t<render_color> render_manager() {

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
	// configure the scene
	int max_object_count = 200;
	
	memory_t<hittable<T>*> lists(max_object_count);
	memory_t<hittable<T>*> world(max_object_count);
	lists.allocate_memory(max_object_count * sizeof(hittable<T>*));
	world.allocate_memory(max_object_count * sizeof(hittable<T>*));
	two_spheres << <1, 1 >> > (lists.data(), world.data());

	// configure the rendering
	size_t size = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * sizeof(render_color);
	memory_t<render_color> device_ptr(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT);
	device_ptr.allocate_memory(size);
	dim3 grid(
		(uint32_t)ceil((T)DEFAULT_IMAGE_WIDTH / NUM_THREADS_MIN), 
		(uint32_t)ceil((T)DEFAULT_IMAGE_HEIGHT / NUM_THREADS_MIN)
	);
	dim3 block(NUM_THREADS_MIN, NUM_THREADS_MIN);

	
	renderer << < grid, block >> > (device_ptr.data(), world.data());	
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time taken to render the image: " 
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() 
		<< " ms" << std::endl;
	
	return device_ptr; 
}

PE_END

#endif /* D67A76C9_4D71_4084_813D_2E63C21A2759 */
