#ifndef MGPU_CUDA_HELPER
#define MGPU_CUDA_HELPER

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/gl.h>
#include <iostream>

void printCudaErr();
void checkHostAlloc(void ** ptr, size_t size, unsigned int flags);
void checkMapBufferObj(void** ptr, GLuint id);
void checkMapBufferObjAsync(void** ptr, GLuint id, cudaStream_t & stream);
void checkUnmapBufferObj(GLuint id);
void checkUnmapBufferObjAsync(GLuint id, cudaStream_t & stream);
void checkRegBufferObj(GLuint id);
void checkUnregBufferObj(GLuint id);

#endif
