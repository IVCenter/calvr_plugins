/**
 * @file CudaHelper.h
 * Contains functions to perform some cuda operations 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef MGPU_CUDA_HELPER
#define MGPU_CUDA_HELPER

#ifdef WIN32
#include <Windows.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <iostream>

/// Check for and print the last cuda error
void printCudaErr();

/// Allocate cuda host pinned memory for async functions
void checkHostAlloc(void ** ptr, size_t size, unsigned int flags);

/// Map an opengl buffer object
void checkMapBufferObj(void** ptr, GLuint id);

/// Map an opengl buffer object asynchronously 
void checkMapBufferObjAsync(void** ptr, GLuint id, cudaStream_t & stream);

/// Unmap an opengl buffer object
void checkUnmapBufferObj(GLuint id);

/// Unmap an opengl buffer object asynchronously 
void checkUnmapBufferObjAsync(GLuint id, cudaStream_t & stream);

/// Register an opengl buffer object
void checkRegBufferObj(GLuint id);

/// Unregister an opengl buffer object
void checkUnregBufferObj(GLuint id);

// 3.2 sdk

/**
 * Class that handles cuda interation with an opengl texture
 */
class CudaGLImage
{
    public:
        CudaGLImage(GLuint id, GLenum target);

        /// register the texture with cuda
        void registerImage(unsigned int flags);
        /// unregister the texture with cuda
        void unregisterImage();

        /// set flag to use when mapping the texture
        void setMapFlags(unsigned int flags);

        /// map texture in cuda address space
        void map();
        /// unmap texture in cuda address space
        void unmap();

        /// get pointer to texture as a cuda array pointer
        cudaArray * getPointer();
    protected:
        GLuint _id;         ///< texture id
        GLenum _target;     ///< texture type

        cudaGraphicsResource * _resource;   ///< cuda resource handle
};

#endif
