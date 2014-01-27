#ifndef CUDA_VOLUME_H
#define CUDA_VOLUME_H

#include <cuda.h>
#include <cuda_runtime.h>

void setViewerInfo(void * position, void * direction);

void launchPreSort(float3 * points, uint4 * indices, int numTets, float * dist, float * slope1, float * slope2, uint3 * preSortInd);
void launchDistClear(float * dist, int entries);
void launchSort(int entries, float * dist, float * slope1, float * slope2, uint3 * preSortInd, uint3 * finalInd, int * firstIndex);

__global__ void preSort(float3 * points, uint4 * indices, int numTets, float * dist, float * slope1, float * slope2, uint3 * preSortInd);
__global__ void distClear(float * dist, int entries);
__global__ void distSort(int entries, float * dist, float * slope1, float * slope2, uint3 * preSortInd, uint3 * finalInd, int * firstIndex);
#endif
