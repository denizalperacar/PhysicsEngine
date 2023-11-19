#ifndef D6A2A673_44D0_4C86_A48F_94ADBEC13E87
#define D6A2A673_44D0_4C86_A48F_94ADBEC13E87

#include "pch.h"

PE_BEGIN

void printGLFWInfo(GLFWwindow* w){
	int p = glfwGetWindowAttrib(w, GLFW_OPENGL_PROFILE);
	std::string version = glfwGetVersionString();
	std::string opengl_profile = "";
	if(p == GLFW_OPENGL_COMPAT_PROFILE){
		opengl_profile = "OpenGL Compatibility Profile";
	}
	else if (p == GLFW_OPENGL_CORE_PROFILE){
		opengl_profile = "OpenGL Core Profile";
	}
	printf("GLFW: %s \n", version.c_str());
	printf("GLFW: %s %i \n", opengl_profile.c_str(), p);
}

class glfw_window{

public:

  glfw_window(){
    init_glfw();
    // window = glfwGetCurrentContext();
    printGLFWInfo(window);
  }

  void init_glfw() {
    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, "CUDA-OpenGL Interop", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);
  }


  void display(uchar4*& device_data) {
    // Map the CUDA resource to get a device pointer
    uchar4* cudaPtr;
    size_t size;

    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&cudaPtr, &size, cudaResource);

    // Copy the data from device_data to the OpenGL texture
    size_t width = DEFAULT_IMAGE_WIDTH;
    size_t height = DEFAULT_IMAGE_HEIGHT;

	  cudaMemcpy(cudaPtr, device_data, DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * sizeof(uchar4), PE_DTD);
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

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  GLFWwindow* window;

};

PE_END


#endif /* D6A2A673_44D0_4C86_A48F_94ADBEC13E87 */
