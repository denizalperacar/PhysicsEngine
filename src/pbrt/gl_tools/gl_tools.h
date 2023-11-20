#ifndef B08A9CCF_3F57_4223_98E2_FE28654F519B
#define B08A9CCF_3F57_4223_98E2_FE28654F519B

#include "pch.h"

PE_BEGIN

void printGlewInfo(){
	printf("GLEW: Glew version: %s \n", glewGetString(GLEW_VERSION));
}

void printGLInfo(){
	printf("OpenGL: GL version: %s \n", glGetString(GL_VERSION));
	printf("OpenGL: GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("OpenGL: Vendor: %s\n", glGetString(GL_VENDOR));
}


PE_END

#endif /* B08A9CCF_3F57_4223_98E2_FE28654F519B */
