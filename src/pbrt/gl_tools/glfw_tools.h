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


PE_END


#endif /* D6A2A673_44D0_4C86_A48F_94ADBEC13E87 */
