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

	// memory_t<uchar4> image(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT);

	uchar4* image;
	cudaMalloc(&image, DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * sizeof(uchar4));
	cudaErr("Image Malloc: ")
	

	dim3 grid(
		(uint32_t)ceil((T)DEFAULT_IMAGE_WIDTH / NUM_THREADS_MIN), 
		(uint32_t)ceil((T)DEFAULT_IMAGE_HEIGHT / NUM_THREADS_MIN)
	);
	dim3 block(NUM_THREADS_MIN, NUM_THREADS_MIN);

  // update renderer
  renderer<<<grid, block>>>(image, world.data());
	cudaErr("renderer error check: ")
	cudaDeviceSynchronize();
	cudaErr("renderer error check: ")

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time taken to render the image: " 
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() 
		<< " ms" << std::endl;

	glfw_window display_window;
	init_glew();
	init_texture();



	while (!glfwWindowShouldClose(display_window.window)) {
		//display_window.display(image);
		// Map the CUDA resource to get a device pointer
		uchar4* cudaPtr;
		size_t size;

		cudaGraphicsMapResources(1, &cudaResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&cudaPtr, &size, cudaResource);

		cudaDeviceSynchronize();
		cudaMemcpy(cudaPtr, image, DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * sizeof(uchar4), cudaMemcpyDeviceToDevice);
		cudaErr("Memcpy from device to mapped OpenGL texture: ")

		// Unmap the CUDA resource
		cudaGraphicsUnmapResources(1, &cudaResource, 0);

		// Render the texture
		glClear(GL_COLOR_BUFFER_BIT);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, glTexture);

		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(-1.0f, -1.0f);
		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(1.0f, -1.0f);
		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(1.0f, 1.0f);
		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(-1.0f, 1.0f);
		glEnd();

		glfwSwapBuffers(display_window.window);
		glfwPollEvents();
	}

	// Cleanup resources
	cudaGraphicsUnregisterResource(cudaResource);
	glDeleteTextures(1, &glTexture);
	glfwDestroyWindow(display_window.window);
	glfwTerminate();


	
	
	cudaFree(image);
  return 0; 
}

PE_END

#endif /* CEB549EA_CE35_4028_8CAF_2B489F74225E */
