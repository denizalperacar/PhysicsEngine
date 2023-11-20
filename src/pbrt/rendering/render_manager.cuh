#ifndef CEB549EA_CE35_4028_8CAF_2B489F74225E
#define CEB549EA_CE35_4028_8CAF_2B489F74225E

#include "pch.h"
#include "basic_renderer.cuh"
#include <GL/glut.h>
#include <GL/glew.h>
#include "../gl_tools/gl_tools.h"
#include "../gl_tools/glew_tools.h"
#include "../gl_tools/glfw_tools.h"

PE_BEGIN



template <typename T> 
int render_manager(int argc, char** argv) {


	// Initialize GLFW
	if (!glfwInit())
			return -1;

	// Create a window
	GLFWwindow* window = glfwCreateWindow(640, 600, "My Window", NULL, NULL);
	if (!window)
	{
			glfwTerminate();
			return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Set up texture parameters
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Set up texture data
	uchar4* img = nullptr; // Replace with your device memory pointer
	// configure the scene
	int max_object_count = 200;
	
	memory_t<hittable<T>*> lists(max_object_count);
	memory_t<hittable<T>*> world(max_object_count);
	lists.allocate_memory(max_object_count * sizeof(hittable<T>*));
	world.allocate_memory(max_object_count * sizeof(hittable<T>*));
	two_spheres << <1, 1 >> > (lists.data(), world.data());

	dim3 grid(
		(uint32_t)ceil((T)DEFAULT_IMAGE_WIDTH / NUM_THREADS_MIN), 
		(uint32_t)ceil((T)DEFAULT_IMAGE_HEIGHT / NUM_THREADS_MIN)
	);
	dim3 block(NUM_THREADS_MIN, NUM_THREADS_MIN);

  // update renderer
  renderer<<<grid, block>>>(img, world.data());
	cudaErr("renderer error check: ")

	int width = 800; // Replace with your texture width
	int height = 600; // Replace with your texture height
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);

	// Render loop
	while (!glfwWindowShouldClose(window))
	{
			// Clear the screen
			glClear(GL_COLOR_BUFFER_BIT);

			// Draw the texture
			glBegin(GL_QUADS);
			glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
			glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
			glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
			glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
			glEnd();

			// Swap buffers
			glfwSwapBuffers(window);

			// Poll for events
			glfwPollEvents();
	}

	// Clean up
	glfwTerminate();
	return 0;
}

/*
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

	GLuint pbo; // OpenGL Pixel Buffer Object
	cudaGraphicsResource* cudaPboResource; // CUDA-OpenGL interop resource
	GLuint textureID; // OpenGL Texture ID

	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Create PBO
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * sizeof(uchar4), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// Register PBO with CUDA
	cudaGraphicsGLRegisterBuffer(&cudaPboResource, pbo, cudaGraphicsMapFlagsWriteDiscard);



	// Cleanup resources
	cudaGraphicsUnregisterResource(cudaPboResource);
	glDeleteBuffers(1, &pbo);
	glfwDestroyWindow(display_window.window);
	glfwTerminate();

  return 0; 
}
*/

PE_END

#endif /* CEB549EA_CE35_4028_8CAF_2B489F74225E */
