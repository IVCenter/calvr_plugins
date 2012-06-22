#ifndef MGPU_CUDA_HELPER
#define MGPU_CUDA_HELPER

#include <cuda.h>
#include <cuda_runtime.h>
#include <GL/gl.h>
#include <iostream>

void printCudaErr();
void checkHostAlloc(void ** ptr, size_t size, unsigned int flags);
void checkMapBufferObj(void** ptr, GLuint id);
void checkUnmapBufferObj(GLuint id);
void checkRegBufferObj(GLuint id);
void checkUnregBufferObj(GLuint id);

#endif
