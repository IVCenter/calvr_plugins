#ifndef CUDA_LIC_H
#define CUDA_LIC_H

#include <cuda.h>
#include <cuda_runtime.h>

void setVelSurfaceRef(cudaArray* array);
void setOutSurfaceRef(cudaArray* array);
void setPlaneConsts(void * point, void * normal, void * right, void * up, void * rightNorm, void * upNorm, void * basisMat, void * bLength);
void setTexConsts(void * xMin, void * xMax, void * yMin, void * yMax);
void launchVel(uint4 * indices, float3 * verts, float3 * velocity, unsigned int * tetList, int numTets, int width, int height);
void launchLIC(int width, int height, float length, cudaArray * noiseArray);
void launchMakeTetList(unsigned int * tetList, unsigned int * numTets, int totalTets, uint4 * indices, float3 * verts);

__global__ void velTest(int width, int height);
__global__ void velKernel(uint4 * ind, float3 * verts, float3 * velocity, unsigned int * tetList, int numTets, int width, int height, float hwidth, float hheight);
__global__ void licKernel(int width, int height, float length);
__global__ void makeTetListKernel(unsigned int * tetList, unsigned int * numTets, int totalTets, uint4 * indices, float3 * verts);
#endif
