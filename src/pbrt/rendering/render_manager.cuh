#ifndef CEB549EA_CE35_4028_8CAF_2B489F74225E
#define CEB549EA_CE35_4028_8CAF_2B489F74225E

#include "pch.h"
#include "basic_renderer.cuh"
#include "../gl_tools/gl_tools.h"
#include "../gl_tools/glew_tools.h"
#include "../gl_tools/glfw_tools.h"

PE_BEGIN

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}



template <typename T>
int render_manager() {

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();	
	
	// configure the scene
	int max_object_count = 200;
	
	memory_t<hittable<T>*> lists(max_object_count);
	memory_t<hittable<T>*> world(max_object_count);
	lists.allocate_memory(max_object_count * sizeof(hittable<T>*));
	world.allocate_memory(max_object_count * sizeof(hittable<T>*));
	two_spheres << <1, 1 >> > (lists.data(), world.data());

	memory_t<uchar4> image(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT);

	dim3 grid(
		(uint32_t)ceil((T)DEFAULT_IMAGE_WIDTH / NUM_THREADS_MIN), 
		(uint32_t)ceil((T)DEFAULT_IMAGE_HEIGHT / NUM_THREADS_MIN)
	);
	dim3 block(NUM_THREADS_MIN, NUM_THREADS_MIN);

  // update renderer
  renderer<<<grid, block>>>(image.data(), world.data());


	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time taken to render the image: " 
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() 
		<< " ms" << std::endl;
	
  return 0; 
}

PE_END

#endif /* CEB549EA_CE35_4028_8CAF_2B489F74225E */
