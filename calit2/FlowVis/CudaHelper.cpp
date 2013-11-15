/**
 * @file CudaHelper.cpp
 * Contains functions to perform some cuda operations 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#include "CudaHelper.h"

#include <cstdio>

using namespace std;

void printCudaErr()
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        printf("Cuda error: \"%s\"\n", cudaGetErrorString(err));
    }
}

/**
 * @param ptr address of pointer to receive memory block address
 * @param size amount of memory to allocate
 * @param flags hints to pass to cudaHostAlloc function 
 */
void checkHostAlloc(void ** ptr, size_t size, unsigned int flags)
{
    if(cudaSuccess != cudaHostAlloc(ptr, size, flags))
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to allocate host pinned memory size: " << size << std::endl;
    }
}

/**
 * @param ptr address of pointer to receive mapped buffer address
 * @param id opengl id of buffer to be mapped
 */
void checkMapBufferObj(void** ptr, GLuint id)
{
    if (cudaSuccess != cudaGLMapBufferObject(ptr, id)) 
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to map buffer object to device memory" << std::endl;
    }
}

/**
 * @param ptr address of pointer to receive mapped buffer address
 * @param id opengl id of buffer to be mapped
 * @param stream cuda stream to attach this async operation to
 */
void checkMapBufferObjAsync(void** ptr, GLuint id, cudaStream_t & stream)
{
    if (cudaSuccess != cudaGLMapBufferObjectAsync(ptr, id, stream)) 
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to map buffer object to device memory" << std::endl;
    }
}

/**
 * @param id opengl id of buffer to unmap
 */
void checkUnmapBufferObj(GLuint id)
{
    const cudaError_t result = cudaGLUnmapBufferObject(id);
    if (cudaSuccess != result) 
    {
        if (cudaErrorInvalidDevicePointer == result)
	{
            printf("Invalid device pointer:\n");
	}
        printCudaErr();
        printf("CUDA: Failed to umap buffer object from device memory (ID %d)\n", id);
    }
}


/**
 * @param id opengl id of buffer to unmap
 * @param stream cuda stream to attach this async operation to
 */
void checkUnmapBufferObjAsync(GLuint id, cudaStream_t & stream)
{
    const cudaError_t result = cudaGLUnmapBufferObjectAsync(id, stream);
    if (cudaSuccess != result) 
    {
        if (cudaErrorInvalidDevicePointer == result)
	{
            printf("Invalid device pointer:\n");
	}
        printCudaErr();
        printf("CUDA: Failed to umap buffer object from device memory (ID %d)\n", id);
    }
}

/**
 * @param id opengl id of buffer to register with cuda
 */
void checkRegBufferObj(GLuint id)
{
    if (cudaSuccess != cudaGLRegisterBufferObject(id)) 
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to register buffer object" << std::endl;
    }
}

/**
 * @param id opengl id of buffer to unregister with cuda
 */
void checkUnregBufferObj(GLuint id)
{
    if (cudaSuccess != cudaGLUnregisterBufferObject(id)) 
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to unregister buffer object" << std::endl;
    }
}

/**
 * @param id opengl id of texture to work with
 * @param target texture type
 */
CudaGLImage::CudaGLImage(GLuint id, GLenum target)
{
    _id = id;
    _target = target;
}

/**
 * @param flags options to pass to cudaGraphicsGLRegisterImage function
 */
void CudaGLImage::registerImage(unsigned int flags)
{
    const cudaError_t result = cudaGraphicsGLRegisterImage((cudaGraphicsResource**)&_resource, _id, _target, flags);
    if (cudaSuccess != result) 
    {
        if (cudaErrorInvalidDevicePointer == result)
	{
            printf("Invalid device pointer:\n");
	}
        printCudaErr();
        printf("CUDA: Failed to register image (ID %d)\n", _id);
    }
}

void CudaGLImage::unregisterImage()
{
    if(!_resource)
    {
	fprintf(stderr,"Invalid resource handle.\n");
	return;
    }

    const cudaError_t result = cudaGraphicsUnregisterResource(_resource);
    if (cudaSuccess != result) 
    {
        if (cudaErrorInvalidDevicePointer == result)
	{
            printf("Invalid device pointer:\n");
	}
        printCudaErr();
        printf("CUDA: Failed to unregister image (ID %d)\n", _id);
    }
}

/**
 * @param flags options to pass to next image mapping call
 */
void CudaGLImage::setMapFlags(unsigned int flags)
{
    if(!_resource)
    {
	fprintf(stderr,"Invalid resource handle.\n");
	return;
    }

    const cudaError_t result = cudaGraphicsResourceSetMapFlags(_resource, flags);
    if (cudaSuccess != result) 
    {
        if (cudaErrorInvalidDevicePointer == result)
	{
            printf("Invalid device pointer:\n");
	}
        printCudaErr();
        printf("CUDA: Failed to set map flags (ID %d)\n", _id);
    }
}

void CudaGLImage::map()
{
    if(!_resource)
    {
	fprintf(stderr,"Invalid resource handle.\n");
	return;
    }

    const cudaError_t result = cudaGraphicsMapResources(1,&_resource);
    if (cudaSuccess != result) 
    {
        if (cudaErrorInvalidDevicePointer == result)
	{
            printf("Invalid device pointer:\n");
	}
        printCudaErr();
        printf("CUDA: Failed to map image (ID %d)\n", _id);
    }
}

void CudaGLImage::unmap()
{
    if(!_resource)
    {
	fprintf(stderr,"Invalid resource handle.\n");
	return;
    }

    const cudaError_t result = cudaGraphicsUnmapResources(1,&_resource);
    if (cudaSuccess != result) 
    {
        if (cudaErrorInvalidDevicePointer == result)
	{
            printf("Invalid device pointer:\n");
	}
        printCudaErr();
        printf("CUDA: Failed to unmap image (ID %d)\n", _id);
    }
}

/**
 * @return pointer to cuda data structure for mapped image
 */
cudaArray * CudaGLImage::getPointer()
{
    if(!_resource)
    {
	fprintf(stderr,"Invalid resource handle.\n");
	return NULL;
    }

    cudaArray * carray;

    const cudaError_t result = cudaGraphicsSubResourceGetMappedArray(&carray,_resource,0,0);
    if (cudaSuccess != result) 
    {
        if (cudaErrorInvalidDevicePointer == result)
	{
            printf("Invalid device pointer:\n");
	}
        printCudaErr();
        printf("CUDA: Failed to get cudaArray pointer (ID %d)\n", _id);
    }

    return carray;
}
