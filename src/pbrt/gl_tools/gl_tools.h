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


void init_texture() {
    glGenTextures(1, &glTexture);
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // Register the texture with CUDA
    cudaGraphicsGLRegisterImage(&cudaResource, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

PE_END

#endif /* B08A9CCF_3F57_4223_98E2_FE28654F519B */
