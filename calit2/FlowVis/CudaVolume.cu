#include "CudaVolume.h"

#include <iostream>

__constant__ float viewerPos[3];
__constant__ float viewerDir[3];

void setViewerInfo(void * position, void * direction)
{
    cudaMemcpyToSymbol(viewerPos,position,3*sizeof(float));
    cudaMemcpyToSymbol(viewerDir,direction,3*sizeof(float));
}

// TODO: optimize block/grid sizes, final reg counts required

void launchPreSort(float3 * points, uint4 * indices, int numTets, float * dist, float * slope1, float * slope2, uint3 * preSortInd)
{
    dim3 block(256,1,1);
    int griddim = (numTets / 256) + 1;
    //std::cerr << "Grid dim: " << griddim << std::endl;
    dim3 grid(griddim,1,1);

    preSort<<< grid, block>>>(points,indices,numTets,dist,slope1,slope2,preSortInd);
}

void launchDistClear(float * dist, int entries)
{
    dim3 block(256,1,1);
    int griddim = (entries / 256) + 1;
    std::cerr << "Grid dim clear: " << griddim << std::endl;
    dim3 grid(griddim,1,1);

    distClear<<< grid, block>>>(dist,entries);
}

void launchSort(int entries, float * dist, float * slope1, float * slope2, uint3 * preSortInd, uint3 * finalInd, int * firstIndex)
{
    dim3 block(256,1,1);
    int griddim = (entries / 256) + 1;
    std::cerr << "Grid dim sort: " << griddim << std::endl;
    dim3 grid(griddim,1,1);

    distSort<<< grid, block>>>(entries,dist,slope1,slope2,preSortInd,finalInd,firstIndex);
}

__global__ void preSort(float3 * points, uint4 * indices, int numTets, float * dist, float * slope1, float * slope2, uint3 * preSortInd)
{
    int tetid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tetid >= numTets)
    {
	return;
    }

    float4 tetpoints[4];
    *((float3*)&tetpoints[0]) = points[indices[tetid].x];
    *((float3*)&tetpoints[1]) = points[indices[tetid].y];
    *((float3*)&tetpoints[2]) = points[indices[tetid].z];
    *((float3*)&tetpoints[3]) = points[indices[tetid].w];

    // find viewing plane distance
    tetpoints[0].w = (tetpoints[0].x - viewerPos[0]) * viewerDir[0] + (tetpoints[0].y - viewerPos[1]) * viewerDir[1] + (tetpoints[0].z - viewerPos[2]) * viewerDir[2];
    tetpoints[1].w = (tetpoints[1].x - viewerPos[0]) * viewerDir[0] + (tetpoints[1].y - viewerPos[1]) * viewerDir[1] + (tetpoints[1].z - viewerPos[2]) * viewerDir[2];
    tetpoints[2].w = (tetpoints[2].x - viewerPos[0]) * viewerDir[0] + (tetpoints[2].y - viewerPos[1]) * viewerDir[1] + (tetpoints[2].z - viewerPos[2]) * viewerDir[2];
    tetpoints[3].w = (tetpoints[3].x - viewerPos[0]) * viewerDir[0] + (tetpoints[3].y - viewerPos[1]) * viewerDir[1] + (tetpoints[3].z - viewerPos[2]) * viewerDir[2];

    // project points onto plane
    tetpoints[0].x = tetpoints[0].x - viewerDir[0] * tetpoints[0].w;
    tetpoints[0].y = tetpoints[0].y - viewerDir[1] * tetpoints[0].w;
    tetpoints[0].z = tetpoints[0].z - viewerDir[2] * tetpoints[0].w;
    tetpoints[1].x = tetpoints[1].x - viewerDir[0] * tetpoints[1].w;
    tetpoints[1].y = tetpoints[1].y - viewerDir[1] * tetpoints[1].w;
    tetpoints[1].z = tetpoints[1].z - viewerDir[2] * tetpoints[1].w;
    tetpoints[2].x = tetpoints[2].x - viewerDir[0] * tetpoints[2].w;
    tetpoints[2].y = tetpoints[2].y - viewerDir[1] * tetpoints[2].w;
    tetpoints[2].z = tetpoints[2].z - viewerDir[2] * tetpoints[2].w;
    tetpoints[3].x = tetpoints[3].x - viewerDir[0] * tetpoints[3].w;
    tetpoints[3].y = tetpoints[3].y - viewerDir[1] * tetpoints[3].w;
    tetpoints[3].z = tetpoints[3].z - viewerDir[2] * tetpoints[3].w;

    int index = 0;

    // triangle 0 1 3
    unsigned int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 3;

    int offset = 0;
    float mindist = tetpoints[order[0]].w;
    if(tetpoints[order[1]].w < tetpoints[order[0]].w)
    {
	mindist = tetpoints[order[1]].w;
	offset = 1;
    }
    if(tetpoints[order[2]].w < mindist)
    {
	offset = 2;
	mindist = tetpoints[order[2]].w;
    }
    offset = 3 - offset;

    unsigned int ind[3];
    ind[offset%3] = order[0];
    ind[(offset+1)%3] = order[1];
    ind[(offset+2)%3] = order[2];

    float3 vec1, vec2, axis;
    vec1.x = tetpoints[ind[1]].x - tetpoints[ind[0]].x;
    vec1.y = tetpoints[ind[1]].y - tetpoints[ind[0]].y;
    vec1.z = tetpoints[ind[1]].z - tetpoints[ind[0]].z;
    vec2.x = tetpoints[ind[2]].x - tetpoints[ind[0]].x;
    vec2.y = tetpoints[ind[2]].y - tetpoints[ind[0]].y;
    vec2.z = tetpoints[ind[2]].z - tetpoints[ind[0]].z;

    axis.x = vec1.y * vec2.z - vec1.z * vec2.y;
    axis.y = vec1.z * vec2.x - vec1.x * vec2.z;
    axis.z = vec1.x * vec2.y - vec1.y * vec2.x;

    float val = axis.x * viewerDir[0] + axis.y * viewerDir[1] + axis.z * viewerDir[2];

    // is triangle front facing
    if(val > 0.0)
    {
	dist[(3*tetid)+index] = mindist;

	preSortInd[(3*tetid)+index].x = ((unsigned int*)&indices[tetid])[ind[0]];
	preSortInd[(3*tetid)+index].y = ((unsigned int*)&indices[tetid])[ind[1]];
	preSortInd[(3*tetid)+index].z = ((unsigned int*)&indices[tetid])[ind[2]];
	
        float s1, s2;
	s1 = (tetpoints[ind[1]].w - tetpoints[ind[0]].w) / sqrt(vec1.x * vec1.x + vec1.y * vec1.y + vec1.z * vec1.z) ;
	s2 = (tetpoints[ind[2]].w - tetpoints[ind[0]].w) / sqrt(vec2.x * vec2.x + vec2.y * vec2.y + vec2.z * vec2.z) ;

	if(s2 < s1)
	{
	    float temps = s1;
	    s1 = s2;
	    s2 = temps;
	}

	slope1[(3*tetid)+index] = s1;
	slope2[(3*tetid)+index] = s2;

	index++;
    }
}

__global__ void distClear(float * dist, int entries)
{
    dist[blockIdx.x*blockDim.x + threadIdx.x] = 0.0;
}

__global__ void distSort(int entries, float * dist, float * slope1, float * slope2, uint3 * preSortInd, uint3 * finalInd, int * first)
{
    int myIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if(myIndex >= entries)
    {
	return;
    }
    float myDist = dist[myIndex];
    float mySlope1 = slope1[myIndex];
    float mySlope2 = slope2[myIndex];
    if(myDist <= 0.0)
    {
	return;
    }

    int finalIndex = 0;
    int firstIndex = 0;
    for(int i = 0; i < entries; ++i)
    {
	float temp = dist[i];
	if(temp <= 0.0)
	{
	    firstIndex++;
	}
	if(temp > myDist || temp <= 0.0)
	{
	    finalIndex++;
	    continue;
	}
	if(temp == myDist)
	{
	    temp = slope1[i];
	    if(temp > mySlope1)
	    {
		finalIndex++;
		continue;
	    }
	    if(temp == mySlope1)
	    {
		temp = slope2[i];
		if(temp > mySlope2)
		{
		    finalIndex++;
		    continue;
		}
	    }
	}
    }

    if(first && finalIndex == firstIndex)
    {
	*first = firstIndex;
    }

    finalInd[finalIndex].x = preSortInd[myIndex].x;
    finalInd[finalIndex].y = preSortInd[myIndex].y;
    finalInd[finalIndex].z = preSortInd[myIndex].z;
}
