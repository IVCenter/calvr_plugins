#include "CudaHelper.h"

#include <cuda_gl_interop.h>

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

void checkHostAlloc(void ** ptr, size_t size, unsigned int flags)
{
    if(cudaSuccess != cudaHostAlloc(ptr, size, flags))
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to allocate host pinned memory size: " << size << std::endl;
    }
}

void checkMapBufferObj(void** ptr, GLuint id)
{
    if (cudaSuccess != cudaGLMapBufferObject(ptr, id)) 
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to map buffer object to device memory" << std::endl;
    }
}

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

void checkRegBufferObj(GLuint id)
{
    if (cudaSuccess != cudaGLRegisterBufferObject(id)) 
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to register buffer object" << std::endl;
    }
}

void checkUnregBufferObj(GLuint id)
{
    if (cudaSuccess != cudaGLUnregisterBufferObject(id)) 
    {
        printCudaErr();
        std::cerr << "CUDA: Failed to unregister buffer object" << std::endl;
    }
}
