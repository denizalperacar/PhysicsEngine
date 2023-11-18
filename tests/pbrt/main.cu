#include "pch.h"
#include "../../src/pbrt/rendering/basic_renderer.cuh"

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

__global__ void fillTextureKernel(cudaSurfaceObject_t surfaceObj, int width, int height) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Your filling logic here
  uchar4 color = make_uchar4(255, 0, 0, 255);  // Red color, modify as needed

  surf2Dwrite(color, surfaceObj, x * sizeof(uchar4), y);
}

// Your OpenGL texture ID
GLuint cudaImageTexture;
// CUDA Graphics Resource
cudaGraphicsResource_t cudaImageResource;

int main () {

  if (!glfwInit()) {
    // Handle initialization failure
    return -1;
  }

  GLFWwindow* window = glfwCreateWindow(PE::DEFAULT_IMAGE_WIDTH, PE::DEFAULT_IMAGE_HEIGHT, "CUDA-OpenGL Interop", NULL, NULL);

  if (!window) {
    // Handle window creation failure
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, key_callback);

  glewExperimental = GL_TRUE;
  // Initialize GLEW
  if (glewInit() != GLEW_OK) {
    // Handle GLEW initialization failure
    glfwDestroyWindow(window);
    glfwTerminate();
    return -1;
  }

  glGenTextures(1, &cudaImageTexture);
  glBindTexture(GL_TEXTURE_2D, cudaImageTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  cudaGraphicsGLRegisterImage(&cudaImageResource, cudaImageTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);


  // Map CUDA resources
  cudaArray* cudaArray;
  cudaGraphicsMapResources(1, &cudaImageResource, 0);
  cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaImageResource, 0, 0);

  // Map CUDA resources
  // Create CUDA resource descriptor
  cudaResourceDesc cudaArrayResourceDesc;
  memset(&cudaArrayResourceDesc, 0, sizeof(cudaResourceDesc));
  cudaArrayResourceDesc.resType = cudaResourceTypeArray;
  cudaArrayResourceDesc.res.array.array = cudaArray;

  int width = (int) PE::DEFAULT_IMAGE_WIDTH;
  int height = (int) PE::DEFAULT_IMAGE_HEIGHT;

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  cudaSurfaceObject_t surfaceObj;
  cudaCreateSurfaceObject(&surfaceObj, &cudaArrayResourceDesc);

  fillTextureKernel<<<gridSize, blockSize>>>(surfaceObj, width, height);
  // PE::memory_t<PE::render_color> render = PE::render_manager<float>();

  // Main rendering loop
  while (!glfwWindowShouldClose(window)) {
      // Render the OpenGL texture
      glClear(GL_COLOR_BUFFER_BIT);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, cudaImageTexture);

      glBegin(GL_QUADS);
      glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
      glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
      glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
      glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
      glEnd();

      glDisable(GL_TEXTURE_2D);

      glfwSwapBuffers(window);
      glfwPollEvents();
  }

  // Cleanup
  cudaGraphicsUnregisterResource(cudaImageResource);
  glDeleteTextures(1, &cudaImageTexture);

  glfwDestroyWindow(window);
  glfwTerminate();

  cudaDestroySurfaceObject(surfaceObj);
  cudaGraphicsUnregisterResource(cudaImageResource);

  return 0;
}