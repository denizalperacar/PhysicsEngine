#ifndef CEB549EA_CE35_4028_8CAF_2B489F74225E
#define CEB549EA_CE35_4028_8CAF_2B489F74225E

#include "pch.h"
#include "basic_renderer.cuh"

#include "../gl_tools/gl_tools.h"
#include "../gl_tools/glew_tools.h"
#include "../gl_tools/glfw_tools.h"

PE_BEGIN

// [TODO]convet all of this to a class 
#define DELTA 5 // pixel increment for arrow keys

#define TITLE_STRING "flashlight: distance image display app"
int2 loc = {DEFAULT_IMAGE_WIDTH/2, DEFAULT_IMAGE_HEIGHT/2};
bool dragMode = false; // mouse tracking mode

void keyboard(unsigned char key, int x, int y) {
	if (key == 'a') dragMode = !dragMode; // toggle tracking mode
	if (key == 27)  exit(0);
	glutPostRedisplay();
}

void mouseMove(int x, int y) {
	if (dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

void mouseDrag(int x, int y) {
	if (!dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
	if (key == GLUT_KEY_LEFT)  loc.x -= DELTA;
	if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
	if (key == GLUT_KEY_UP)    loc.y -= DELTA;
	if (key == GLUT_KEY_DOWN)  loc.y += DELTA;
	glutPostRedisplay();
}

void printInstructions() {
	printf("flashlight interactions\n");
	printf("a: toggle mouse tracking mode\n");
	printf("arrow keys: move ref location\n");
	printf("esc: close graphics window\n");
}

void render() {

	// configure the scene
	int max_object_count = 200;

	memory_t<hittable<float>*> lists(max_object_count);
	memory_t<hittable<float>*> world(max_object_count);
	lists.allocate_memory(max_object_count * sizeof(hittable<float>*));
	world.allocate_memory(max_object_count * sizeof(hittable<float>*));
	two_spheres << <1, 1 >> > (lists.data(), world.data());

  uchar4 *img = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&img, NULL, cuda_pbo_resource);
  
	dim3 grid(
		(uint32_t)ceil((float)DEFAULT_IMAGE_WIDTH / NUM_THREADS_MIN), 
		(uint32_t)ceil((float)DEFAULT_IMAGE_HEIGHT / NUM_THREADS_MIN)
	);
	dim3 block(NUM_THREADS_MIN, NUM_THREADS_MIN);

	// update renderer
	renderer<<<grid, block>>>(img, world.data());
	cudaErr("renderer error check: ")
	cudaDeviceSynchronize();
	cudaErr("renderer error check: ")

  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0, DEFAULT_IMAGE_HEIGHT);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(DEFAULT_IMAGE_WIDTH, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void display() {
  render();
  drawTexture();
  glutSwapBuffers();
}

void initGLUT(int *argc, char **argv) {
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT);
  glutCreateWindow(TITLE_STRING);
  glewInit();

}

void initPixelBuffer() {
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, 4*DEFAULT_IMAGE_WIDTH*DEFAULT_IMAGE_HEIGHT*sizeof(GLubyte), 0,
               GL_STREAM_DRAW);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                               cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc() {
  if (pbo) {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
  }
}

template <typename T> 
int render_manager(int argc, char** argv) {

	cudaDeviceReset();
	cudaErr("Device Reset: ")

	printInstructions();
	initGLUT(&argc, argv);
	gluOrtho2D(0, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, 0);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(handleSpecialKeypress);
	glutPassiveMotionFunc(mouseMove);
	glutMotionFunc(mouseDrag);
	glutDisplayFunc(display);
	initPixelBuffer();
	glutMainLoop();
	atexit(exitfunc);

	return 0;
}

PE_END

#endif /* CEB549EA_CE35_4028_8CAF_2B489F74225E */
