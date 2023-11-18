#ifndef A40A2EDB_9907_4717_B1B8_A342A590F199
#define A40A2EDB_9907_4717_B1B8_A342A590F199

#include "pch.h"

PE_BEGIN

void init_glew() {
    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Ensure we can use OpenGL 3.3
    if (!GLEW_VERSION_3_3) {
        fprintf(stderr, "OpenGL 3.3 not supported\n");
        exit(EXIT_FAILURE);
    }
}

PE_END

#endif /* A40A2EDB_9907_4717_B1B8_A342A590F199 */
