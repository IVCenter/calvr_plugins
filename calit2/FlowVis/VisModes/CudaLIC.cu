#include "CudaLIC.h"

#include <iostream>
#include <stdio.h>
#include <float.h>

surface<void, cudaSurfaceType2D> velSurface;
__constant__ float3 planePoint;
__constant__ float3 planeNormal;
__constant__ float3 planeRight;
__constant__ float3 planeUp;
__constant__ float3 planeRightNorm;
__constant__ float3 planeUpNorm;
__constant__ float basisLength;

surface<void, cudaSurfaceType2D> outSurface;

texture<float,2> noiseTex;

__constant__ float texXMin;
__constant__ float texXMax;
__constant__ float texYMin;
__constant__ float texYMax;

void setVelSurfaceRef(cudaArray* array)
{
    cudaBindSurfaceToArray(velSurface, array);
}

void setOutSurfaceRef(cudaArray* array)
{
    cudaBindSurfaceToArray(outSurface, array);
}

void setPlaneConsts(void * point, void * normal, void * right, void * up, void * rightNorm, void * upNorm, void * bLength)
{
    cudaMemcpyToSymbol(planePoint,point,sizeof(float3));
    cudaMemcpyToSymbol(planeNormal,normal,sizeof(float3));
    cudaMemcpyToSymbol(planeRight,right,sizeof(float3));
    cudaMemcpyToSymbol(planeUp,up,sizeof(float3));
    cudaMemcpyToSymbol(planeRightNorm,rightNorm,sizeof(float3));
    cudaMemcpyToSymbol(planeUpNorm,upNorm,sizeof(float3));
    cudaMemcpyToSymbol(basisLength,bLength,sizeof(float));
}

void setTexConsts(void * xMin, void * xMax, void * yMin, void * yMax)
{
    cudaMemcpyToSymbol(texXMin,xMin,sizeof(float));
    cudaMemcpyToSymbol(texXMax,xMax,sizeof(float));
    cudaMemcpyToSymbol(texYMin,yMin,sizeof(float));
    cudaMemcpyToSymbol(texYMax,yMax,sizeof(float));
}

void launchVel(uint4 * indices, float3 * verts, float3 * velocity, unsigned int * tetList, int numTets, int width, int height)
{
    //std::cerr << "Running velocity kernel" << std::endl;

    dim3 blockc(8,8,1);
    dim3 gridc(width/8,height/8,1);

    velClear<<< gridc, blockc >>>(width,height);
    cudaThreadSynchronize();

    int threadsPerBlock = 256;
    dim3 block(threadsPerBlock,1,1);

    int critcalPoints = width * height * 1.1;
    int pointsPerTet = critcalPoints * 1.2 / numTets;
    float blocksPerTet = pointsPerTet / (threadsPerBlock * 20.0f);
    
    int griddim;

    bool threadPerTet = false;

    int sharedMemPerBlock = 0;
    int tetsPerBlock = 1;

    if(blocksPerTet >= 1.0f)
    {
	griddim = ((int)blocksPerTet) * numTets;
	sharedMemPerBlock = 24 * sizeof(float);
    }
    else
    {
	float tetsPerBlockf = 1.0f / blocksPerTet;

	// may need to lower limit if reg count goes down
	if(tetsPerBlockf > 64.0f)
	{
	    griddim = (numTets / 256) + 1;
	    threadPerTet = true;
	}
	else if(tetsPerBlockf > 32.0f)
	{
	    griddim = (numTets / 64) + 1;
	    tetsPerBlock = 64;
	}
	else if(tetsPerBlockf > 16.0f)
	{
	    griddim = (numTets / 32) + 1;
	    tetsPerBlock = 32;
	}
	else if(tetsPerBlockf > 8.0f)
	{
	    griddim = (numTets / 16) + 1;
	    tetsPerBlock = 16;
	}
	else if(tetsPerBlockf > 4.0f)
	{
	    griddim = (numTets / 8) + 1;
	    tetsPerBlock = 8;
	}
	else if(tetsPerBlockf > 2.0f)
	{
	    griddim = (numTets / 4) + 1;
	    tetsPerBlock = 4;
	}
	else
	{
	    griddim = (numTets / 2  ) + 1;
	    tetsPerBlock = 2;
	}
	sharedMemPerBlock = 24 * sizeof(float) * tetsPerBlock;
    }

    //std::cerr << "NumTets: " << numTets << " Grid dim: " << griddim << std::endl;
    dim3 grid(griddim,1,1);

    if(threadPerTet)
    {
	//std::cerr << "Using single thread kernel" << std::endl;
	velKernel<<< grid, block >>>(indices,verts,velocity,tetList,numTets,width,height,width/2.0,height/2.0);
    }
    else
    {
	//std::cerr << "Using multi thread kernel, blocksPerTet: " << (int)blocksPerTet << " tetsPerBlock: " << tetsPerBlock << " smem: " << sharedMemPerBlock << std::endl;
	velSplitKernel<<< grid, block, sharedMemPerBlock >>>(indices,verts,velocity,tetList,numTets,(int)blocksPerTet,tetsPerBlock,width,height,width/2.0,height/2.0);
    }
}

void launchLIC(int width, int height, float length, cudaArray * noiseArray)
{
    //std::cerr << "Running LIC kernel" << std::endl;
    cudaBindTextureToArray(noiseTex,noiseArray);

    dim3 blockc(8,8,1);
    dim3 gridc(width/8,height/8,1);

    licKernel<<< gridc, blockc >>>(width,height,length);
    cudaThreadSynchronize();

    cudaUnbindTexture(noiseTex);
}

void launchMakeTetList(unsigned int * tetList, unsigned int * numTets, int totalTets, uint4 * indices, float3 * verts)
{
    dim3 block(256,1,1);
    int griddim = (totalTets / 256) + 1;
    // 65535 is dimension limit for compute version 2.x and below
    int gridy = (griddim / 65535) + 1;
    int gridx = griddim / gridy;
    //std::cerr << "Grid dim: " << gridx << " " << gridy << std::endl;
    dim3 grid(gridx,gridy,1);

    makeTetListKernel<<< grid, block >>>(tetList,numTets,totalTets,indices,verts);
}

__global__ void velClear(int width, int height) 
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        unsigned short data[2];
	data[0] = __float2half_rn(0.0);
	data[1] = __float2half_rn(0.0);
        surf2Dwrite(*((uchar4*)data), velSurface, x * 4, y);
    }
}

#define VEL_EPS 0.001f

__global__ void velKernel(uint4 * ind, float3 * verts, float3 * velocity, unsigned int * tetList, int numTets, int width, int height, float hwidth, float hheight)
{
    int tetid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tetid >= numTets)
    {
	return;
    }

    tetid = tetList[tetid];

    float4 tetpoints[4];
    *((float3*)&tetpoints[0]) = verts[ind[tetid].x];
    *((float3*)&tetpoints[1]) = verts[ind[tetid].y];
    *((float3*)&tetpoints[2]) = verts[ind[tetid].z];
    *((float3*)&tetpoints[3]) = verts[ind[tetid].w];

    // find viewing plane distance
    tetpoints[0].w = (tetpoints[0].x - planePoint.x) * planeNormal.x + (tetpoints[0].y - planePoint.y) * planeNormal.y + (tetpoints[0].z - planePoint.z) * planeNormal.z;
    tetpoints[1].w = (tetpoints[1].x - planePoint.x) * planeNormal.x + (tetpoints[1].y - planePoint.y) * planeNormal.y + (tetpoints[1].z - planePoint.z) * planeNormal.z;
    tetpoints[2].w = (tetpoints[2].x - planePoint.x) * planeNormal.x + (tetpoints[2].y - planePoint.y) * planeNormal.y + (tetpoints[2].z - planePoint.z) * planeNormal.z;
    tetpoints[3].w = (tetpoints[3].x - planePoint.x) * planeNormal.x + (tetpoints[3].y - planePoint.y) * planeNormal.y + (tetpoints[3].z - planePoint.z) * planeNormal.z;

    // project points onto plane and find basis values
    float3 projpoint;
    float2 basisMin;
    float2 basisMax;
    float tempx;
    projpoint.x = tetpoints[0].x - (planeNormal.x * tetpoints[0].w) - planePoint.x;
    projpoint.y = tetpoints[0].y - (planeNormal.y * tetpoints[0].w) - planePoint.y;
    projpoint.z = tetpoints[0].z - (planeNormal.z * tetpoints[0].w) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = basisMax.x = tempx;
    basisMin.y = basisMax.y = projpoint.y;
    
    projpoint.x = tetpoints[1].x - (planeNormal.x * tetpoints[1].w) - planePoint.x;
    projpoint.y = tetpoints[1].y - (planeNormal.y * tetpoints[1].w) - planePoint.y;
    projpoint.z = tetpoints[1].z - (planeNormal.z * tetpoints[1].w) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    projpoint.x = tetpoints[2].x - (planeNormal.x * tetpoints[2].w) - planePoint.x;
    projpoint.y = tetpoints[2].y - (planeNormal.y * tetpoints[2].w) - planePoint.y;
    projpoint.z = tetpoints[2].z - (planeNormal.z * tetpoints[2].w) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    projpoint.x = tetpoints[3].x - (planeNormal.x * tetpoints[3].w) - planePoint.x;
    projpoint.y = tetpoints[3].y - (planeNormal.y * tetpoints[3].w) - planePoint.y;
    projpoint.z = tetpoints[3].z - (planeNormal.z * tetpoints[3].w) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    /*basisMin.x = ceilf(basisMin.x - 0.5f) + 0.5f;
    basisMin.y = ceilf(basisMin.y - 0.5f) + 0.5f;
    basisMax.x = floorf(basisMax.x - 0.5f) + 0.500001f;
    basisMax.y = floorf(basisMax.y - 0.5f) + 0.500001f;*/

    basisMin.x = floorf(basisMin.x - 0.5f) - 0.5f;
    basisMin.y = floorf(basisMin.y - 0.5f) - 0.5f;
    basisMax.x = ceilf(basisMax.x + 0.5f) + 0.500001f;
    basisMax.y = ceilf(basisMax.y + 0.5f) + 0.500001f;

    if(basisMin.x > basisMax.x || basisMin.y > basisMax.y)
    {
	// no critical points in tet
	return;
    }

    //find if tet is outside the basis bounds
    if(basisMin.x > texXMax || basisMax.x < texXMin || basisMin.y > texYMax || basisMax.y < texYMin)
    {
	return;
    }

    // create matrix to solve for barycentric coords

    // create matrix rows to inverse
    tetpoints[0].x = tetpoints[0].x - tetpoints[3].x;
    tetpoints[0].y = tetpoints[0].y - tetpoints[3].y;
    tetpoints[0].z = tetpoints[0].z - tetpoints[3].z;
    tetpoints[1].x = tetpoints[1].x - tetpoints[3].x;
    tetpoints[1].y = tetpoints[1].y - tetpoints[3].y;
    tetpoints[1].z = tetpoints[1].z - tetpoints[3].z;
    tetpoints[2].x = tetpoints[2].x - tetpoints[3].x;
    tetpoints[2].y = tetpoints[2].y - tetpoints[3].y;
    tetpoints[2].z = tetpoints[2].z - tetpoints[3].z;
    
    // matrix determinant
    float det = tetpoints[0].x*tetpoints[1].y*tetpoints[2].z + tetpoints[1].x*tetpoints[2].y*tetpoints[0].z + tetpoints[2].x*tetpoints[0].y*tetpoints[1].z - tetpoints[2].x*tetpoints[1].y*tetpoints[0].z - tetpoints[1].x*tetpoints[0].y*tetpoints[2].z - tetpoints[0].x*tetpoints[2].y*tetpoints[1].z;

    // invert
    float tetMat[9];
    tetMat[0] = (tetpoints[2].z*tetpoints[1].y - tetpoints[1].z*tetpoints[2].y) / det;
    tetMat[1] = (tetpoints[2].z*tetpoints[1].x - tetpoints[1].z*tetpoints[2].x) / -det;
    tetMat[2] = (tetpoints[2].y*tetpoints[1].x - tetpoints[1].y*tetpoints[2].x) / det;
    tetMat[3] = (tetpoints[2].z*tetpoints[0].y - tetpoints[0].z*tetpoints[2].y) / -det;
    tetMat[4] = (tetpoints[2].z*tetpoints[0].x - tetpoints[0].z*tetpoints[2].x) / det;
    tetMat[5] = (tetpoints[2].y*tetpoints[0].x - tetpoints[0].y*tetpoints[2].x) / -det;
    tetMat[6] = (tetpoints[1].z*tetpoints[0].y - tetpoints[0].z*tetpoints[1].y) / det;
    tetMat[7] = (tetpoints[1].z*tetpoints[0].x - tetpoints[0].z*tetpoints[1].x) / -det;
    tetMat[8] = (tetpoints[1].y*tetpoints[0].x - tetpoints[0].y*tetpoints[1].x) / det;

    // get tet point velocities
     float3 tetVel[4];
    tetVel[0] = velocity[ind[tetid].x];
    tetVel[1] = velocity[ind[tetid].y];
    tetVel[2] = velocity[ind[tetid].z];
    tetVel[3] = velocity[ind[tetid].w];

    // process all critical points
    for(float i = basisMin.x; i <= basisMax.x; i = i + 1.0f)
    {
	for(float j = basisMin.y; j <= basisMax.y; j = j + 1.0f)
	{
	    // find barycentric coords
	    float3 tempPoint;

	    tempPoint.x = i * planeUp.x + j * planeRight.x + planePoint.x - tetpoints[3].x;
	    tempPoint.y = i * planeUp.y + j * planeRight.y + planePoint.y - tetpoints[3].y;
	    tempPoint.z = i * planeUp.z + j * planeRight.z + planePoint.z - tetpoints[3].z;

	    float4 coords;
	    coords.x = tempPoint.x * tetMat[0] + tempPoint.y * tetMat[1] + tempPoint.z * tetMat[2];
	    coords.y = tempPoint.x * tetMat[3] + tempPoint.y * tetMat[4] + tempPoint.z * tetMat[5];
	    coords.z = tempPoint.x * tetMat[6] + tempPoint.y * tetMat[7] + tempPoint.z * tetMat[8];
	    coords.w = 1.0f - coords.x - coords.y - coords.z;

	    if(coords.x > 1.0f + VEL_EPS || coords.x < 0.0f - VEL_EPS || coords.y > 1.0f + VEL_EPS || coords.y < 0.0f - VEL_EPS || coords.z > 1.0f + VEL_EPS || coords.z < 0.0f - VEL_EPS || coords.w > 1.0f + VEL_EPS || coords.w < 0.0f - VEL_EPS)
	    {	
		continue;
	    }

	    int2 texelIndex;
	    texelIndex.x = lrintf((floorf(j) + hwidth));
	    texelIndex.y = lrintf(floor(i) + hheight);
	    if(texelIndex.x >= 0 && texelIndex.x < width && texelIndex.y >= 0 && texelIndex.y < height)
	    {
		// find point velocity
		float3 myVelocity;
		myVelocity.x = coords.x * tetVel[0].x + coords.y * tetVel[1].x + coords.z * tetVel[2].x + coords.w * tetVel[3].x;
		myVelocity.y = coords.x * tetVel[0].y + coords.y * tetVel[1].y + coords.z * tetVel[2].y + coords.w * tetVel[3].y;
		myVelocity.z = coords.x * tetVel[0].z + coords.y * tetVel[1].z + coords.z * tetVel[2].z + coords.w * tetVel[3].z;

		// also temp x
		float mag;
		mag = planeRightNorm.x * myVelocity.x + planeRightNorm.y * myVelocity.y + planeRightNorm.z * myVelocity.z;
		myVelocity.y = planeUpNorm.x * myVelocity.x + planeUpNorm.y * myVelocity.y + planeUpNorm.z * myVelocity.z;
		myVelocity.x = mag;

		// normalize
		mag = myVelocity.x * myVelocity.x + myVelocity.y * myVelocity.y;
		mag = sqrt(mag);
		if(mag > 0.0f)
		{
		    myVelocity.x = myVelocity.x / mag;
		    myVelocity.y = myVelocity.y / mag;
		}

		unsigned short data[2];
		data[0] = __float2half_rn(myVelocity.x);
		data[1] = __float2half_rn(myVelocity.y);
		surf2Dwrite(*((uchar4*)data), velSurface, texelIndex.x * 4, texelIndex.y);
	    }
	}
    }
}

__global__ void velSplitKernel(uint4 * ind, float3 * verts, float3 * velocity, unsigned int * tetList, int numTets, int blocksPerTet, int tetsPerBlock, int width, int height, float hwidth, float hheight)
{
    extern __shared__ float3 tetData[];

    int myOffset;
    int blockOffset;
    int threadOffset;
    int threadsPerTet;

    if(blocksPerTet == 0)
    {
	blockOffset = blockIdx.x * tetsPerBlock;
	myOffset = threadIdx.x / (blockDim.x / tetsPerBlock);
	threadOffset = threadIdx.x % (blockDim.x / tetsPerBlock);
	threadsPerTet = (blockDim.x / tetsPerBlock);
    }
    else
    {
	blockOffset = blockIdx.x / blocksPerTet;
	myOffset = 0;
	threadOffset = (blockIdx.x % blocksPerTet) * blockDim.x + threadIdx.x;
	threadsPerTet = blockDim.x * blocksPerTet;
    }

    if(blockOffset + myOffset >= numTets)
    {
	return;
    }

    int tetid = tetList[blockOffset + myOffset];

    myOffset = myOffset * 4;

    // read tet data
    if(threadOffset == 0 || threadIdx.x == 0)
    {
	int toffset = myOffset;
	tetData[toffset] = verts[ind[tetid].x];
	tetData[toffset+1] = verts[ind[tetid].y];
	tetData[toffset+2] = verts[ind[tetid].z];
	tetData[toffset+3] = verts[ind[tetid].w];

	toffset = toffset + 4 * tetsPerBlock;
	tetData[toffset] = velocity[ind[tetid].x];
	tetData[toffset+1] = velocity[ind[tetid].y];
	tetData[toffset+2] = velocity[ind[tetid].z];
	tetData[toffset+3] = velocity[ind[tetid].w];
    }

    __syncthreads();

    float dist[4];

    // find viewing plane distance
    dist[0] = (tetData[myOffset+0].x - planePoint.x) * planeNormal.x + (tetData[myOffset+0].y - planePoint.y) * planeNormal.y + (tetData[myOffset+0].z - planePoint.z) * planeNormal.z;
    dist[1] = (tetData[myOffset+1].x - planePoint.x) * planeNormal.x + (tetData[myOffset+1].y - planePoint.y) * planeNormal.y + (tetData[myOffset+1].z - planePoint.z) * planeNormal.z;
    dist[2] = (tetData[myOffset+2].x - planePoint.x) * planeNormal.x + (tetData[myOffset+2].y - planePoint.y) * planeNormal.y + (tetData[myOffset+2].z - planePoint.z) * planeNormal.z;
    dist[3] = (tetData[myOffset+3].x - planePoint.x) * planeNormal.x + (tetData[myOffset+3].y - planePoint.y) * planeNormal.y + (tetData[myOffset+3].z - planePoint.z) * planeNormal.z;

    // project points onto plane and find basis values
    float3 projpoint;
    float2 basisMin;
    float2 basisMax;
    float tempx;
    projpoint.x = tetData[myOffset+0].x - (planeNormal.x * dist[0]) - planePoint.x;
    projpoint.y = tetData[myOffset+0].y - (planeNormal.y * dist[0]) - planePoint.y;
    projpoint.z = tetData[myOffset+0].z - (planeNormal.z * dist[0]) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = basisMax.x = tempx;
    basisMin.y = basisMax.y = projpoint.y;
    
    projpoint.x = tetData[myOffset+1].x - (planeNormal.x * dist[1]) - planePoint.x;
    projpoint.y = tetData[myOffset+1].y - (planeNormal.y * dist[1]) - planePoint.y;
    projpoint.z = tetData[myOffset+1].z - (planeNormal.z * dist[1]) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    projpoint.x = tetData[myOffset+2].x - (planeNormal.x * dist[2]) - planePoint.x;
    projpoint.y = tetData[myOffset+2].y - (planeNormal.y * dist[2]) - planePoint.y;
    projpoint.z = tetData[myOffset+2].z - (planeNormal.z * dist[2]) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    projpoint.x = tetData[myOffset+3].x - (planeNormal.x * dist[3]) - planePoint.x;
    projpoint.y = tetData[myOffset+3].y - (planeNormal.y * dist[3]) - planePoint.y;
    projpoint.z = tetData[myOffset+3].z - (planeNormal.z * dist[3]) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    // area larger than needed for rounding errors
    basisMin.x = floorf(basisMin.x - 0.5f) - 0.5f;
    basisMin.y = floorf(basisMin.y - 0.5f) - 0.5f;
    basisMax.x = ceilf(basisMax.x + 0.5f) + 0.500001f;
    basisMax.y = ceilf(basisMax.y + 0.5f) + 0.500001f;

    if(basisMin.x > basisMax.x || basisMin.y > basisMax.y)
    {
	// no critical points in tet
	return;
    }

    //find if tet is outside the basis bounds
    if(basisMin.x > texXMax || basisMax.x < texXMin || basisMin.y > texYMax || basisMax.y < texYMin)
    {
	return;
    }

    // create matrix to solve for barycentric coords

    // create matrix rows to inverse
    tetData[myOffset+0].x = tetData[myOffset+0].x - tetData[myOffset+3].x;
    tetData[myOffset+0].y = tetData[myOffset+0].y - tetData[myOffset+3].y;
    tetData[myOffset+0].z = tetData[myOffset+0].z - tetData[myOffset+3].z;
    tetData[myOffset+1].x = tetData[myOffset+1].x - tetData[myOffset+3].x;
    tetData[myOffset+1].y = tetData[myOffset+1].y - tetData[myOffset+3].y;
    tetData[myOffset+1].z = tetData[myOffset+1].z - tetData[myOffset+3].z;
    tetData[myOffset+2].x = tetData[myOffset+2].x - tetData[myOffset+3].x;
    tetData[myOffset+2].y = tetData[myOffset+2].y - tetData[myOffset+3].y;
    tetData[myOffset+2].z = tetData[myOffset+2].z - tetData[myOffset+3].z;
    
    // matrix determinant
    float det = tetData[myOffset+0].x*tetData[myOffset+1].y*tetData[myOffset+2].z + tetData[myOffset+1].x*tetData[myOffset+2].y*tetData[myOffset+0].z + tetData[myOffset+2].x*tetData[myOffset+0].y*tetData[myOffset+1].z - tetData[myOffset+2].x*tetData[myOffset+1].y*tetData[myOffset+0].z - tetData[myOffset+1].x*tetData[myOffset+0].y*tetData[myOffset+2].z - tetData[myOffset+0].x*tetData[myOffset+2].y*tetData[myOffset+1].z;

    // invert
    float tetMat[9];
    tetMat[0] = (tetData[myOffset+2].z*tetData[myOffset+1].y - tetData[myOffset+1].z*tetData[myOffset+2].y) / det;
    tetMat[1] = (tetData[myOffset+2].z*tetData[myOffset+1].x - tetData[myOffset+1].z*tetData[myOffset+2].x) / -det;
    tetMat[2] = (tetData[myOffset+2].y*tetData[myOffset+1].x - tetData[myOffset+1].y*tetData[myOffset+2].x) / det;
    tetMat[3] = (tetData[myOffset+2].z*tetData[myOffset+0].y - tetData[myOffset+0].z*tetData[myOffset+2].y) / -det;
    tetMat[4] = (tetData[myOffset+2].z*tetData[myOffset+0].x - tetData[myOffset+0].z*tetData[myOffset+2].x) / det;
    tetMat[5] = (tetData[myOffset+2].y*tetData[myOffset+0].x - tetData[myOffset+0].y*tetData[myOffset+2].x) / -det;
    tetMat[6] = (tetData[myOffset+1].z*tetData[myOffset+0].y - tetData[myOffset+0].z*tetData[myOffset+1].y) / det;
    tetMat[7] = (tetData[myOffset+1].z*tetData[myOffset+0].x - tetData[myOffset+0].z*tetData[myOffset+1].x) / -det;
    tetMat[8] = (tetData[myOffset+1].y*tetData[myOffset+0].x - tetData[myOffset+0].y*tetData[myOffset+1].x) / det;


    int2 basisRange;
    basisRange.x = (int)(basisMax.x - basisMin.x);
    basisRange.y = (int)(basisMax.y - basisMin.y);
    
    int velOffset = myOffset + 4 * tetsPerBlock;
    while(1)
    {
	float2 basis;
	basis.x = basisMin.x + (float)(threadOffset % basisRange.x);
	basis.y = basisMin.y + (float)(threadOffset / basisRange.x);

	threadOffset = threadOffset + threadsPerTet;

	if(basis.y > basisMax.y)
	{
	    break;
	}

	// find barycentric coords
	float3 tempPoint;

	tempPoint.x = basis.x * planeUp.x + basis.y * planeRight.x + planePoint.x - tetData[myOffset+3].x;
	tempPoint.y = basis.x * planeUp.y + basis.y * planeRight.y + planePoint.y - tetData[myOffset+3].y;
	tempPoint.z = basis.x * planeUp.z + basis.y * planeRight.z + planePoint.z - tetData[myOffset+3].z;

	float4 coords;
	coords.x = tempPoint.x * tetMat[0] + tempPoint.y * tetMat[1] + tempPoint.z * tetMat[2];
	coords.y = tempPoint.x * tetMat[3] + tempPoint.y * tetMat[4] + tempPoint.z * tetMat[5];
	coords.z = tempPoint.x * tetMat[6] + tempPoint.y * tetMat[7] + tempPoint.z * tetMat[8];
	coords.w = 1.0f - coords.x - coords.y - coords.z;

	if(coords.x > 1.0f + VEL_EPS || coords.x < 0.0f - VEL_EPS || coords.y > 1.0f + VEL_EPS || coords.y < 0.0f - VEL_EPS || coords.z > 1.0f + VEL_EPS || coords.z < 0.0f - VEL_EPS || coords.w > 1.0f + VEL_EPS || coords.w < 0.0f - VEL_EPS)
	{	
	    continue;
	}

	int2 texelIndex;
	texelIndex.x = lrintf((floorf(basis.y) + hwidth));
	texelIndex.y = lrintf(floor(basis.x) + hheight);
	if(texelIndex.x >= 0 && texelIndex.x < width && texelIndex.y >= 0 && texelIndex.y < height)
	{
	    // find point velocity
	    float3 myVelocity;
	    myVelocity.x = coords.x * tetData[velOffset+0].x + coords.y * tetData[velOffset+1].x + coords.z * tetData[velOffset+2].x + coords.w * tetData[velOffset+3].x;
	    myVelocity.y = coords.x * tetData[velOffset+0].y + coords.y * tetData[velOffset+1].y + coords.z * tetData[velOffset+2].y + coords.w * tetData[velOffset+3].y;
	    myVelocity.z = coords.x * tetData[velOffset+0].z + coords.y * tetData[velOffset+1].z + coords.z * tetData[velOffset+2].z + coords.w * tetData[velOffset+3].z;

	    // also temp x
	    float mag;
	    mag = planeRightNorm.x * myVelocity.x + planeRightNorm.y * myVelocity.y + planeRightNorm.z * myVelocity.z;
	    myVelocity.y = planeUpNorm.x * myVelocity.x + planeUpNorm.y * myVelocity.y + planeUpNorm.z * myVelocity.z;
	    myVelocity.x = mag;

	    // normalize
	    mag = myVelocity.x * myVelocity.x + myVelocity.y * myVelocity.y;
	    mag = sqrt(mag);
	    if(mag > 0.0f)
	    {
		myVelocity.x = myVelocity.x / mag;
		myVelocity.y = myVelocity.y / mag;
	    }

	    unsigned short data[2];
	    data[0] = __float2half_rn(myVelocity.x);
	    data[1] = __float2half_rn(myVelocity.y);
	    surf2Dwrite(*((uchar4*)data), velSurface, texelIndex.x * 4, texelIndex.y);
	}
    }
}

#define ROUNDING_ADDITION 0.0001f

__global__ void licKernel(int width, int height, float length)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float pixelVal = 0.0f;
    float pixelDeb = 0.0f;
    // one thread per pixel
    if (x < width && y < height)
    {
	float2 position;
	float2 weightTotal;

	// init to pixel center
	position.x = x + 0.5f;
	position.y = y + 0.5f;

	weightTotal.x = weightTotal.y = 0.0f;

	int maxIterations = length * 100.0f;
	// forward path
	for(int i = 0; i < maxIterations; ++i)
	{
	    unsigned short data[2];
	    // Read from input surface
	    surf2Dread(((uchar4*)data), velSurface, ((int)position.x) * 4, ((int)position.y));
	    float2 velocity;
	    velocity.x = __half2float(data[0]);
	    velocity.y = __half2float(data[1]);

	    // find delta s
	    float top,bottom,left,right;
	    if(velocity.y == 0.0f)
	    {
		top = bottom = FLT_MAX;
	    }
	    else
	    {
		top = (ceilf(position.y) - position.y) / velocity.y;
		bottom = (floorf(position.y) - position.y) / velocity.y;
	    }

	    if(velocity.x == 0.0f)
	    {
		left = right = FLT_MAX;
	    }
	    else
	    {
		right = (ceilf(position.x) - position.x) / velocity.x;
		left = (floorf(position.x) - position.x) / velocity.x;
	    }
	    
	    float deltas = FLT_MAX;
	    
	    if(top > 0.0f && top < deltas)
	    {
		deltas = top;
	    }
	    if(bottom > 0.0f && bottom < deltas)
	    {
		deltas = bottom;
	    }
	    if(right > 0.0f && right < deltas)
	    {
		deltas = right;
	    }
	    if(left > 0.0f && left < deltas)
	    {
		deltas = left;
	    }

	    // zero velocity, going nowhere
	    if(deltas == FLT_MAX)
	    {
		break;
	    }

	    // make sure we hit the next cell
	    deltas = deltas + ROUNDING_ADDITION;

	    // add weight/value

	    // end of line
	    if(weightTotal.x + deltas > length)
	    {
		break;
	    }

	    weightTotal.x = weightTotal.x + deltas;

	    // add weighted noise
	    pixelVal = pixelVal + (deltas * tex2D(noiseTex,(int)position.x,(int)position.y));
	    //printf("PixelVal: %f\n",pixelVal);

	    // find next point
	    position.x = position.x + deltas * velocity.x;
	    position.y = position.y + deltas * velocity.y;

	    // out of grid
	    if(position.x < 0.0f || position.x >= width || position.y < 0.0f || position.y >= height)
	    {
		break;
	    }
	}

	position.x = x + 0.5f;
	position.y = y + 0.5f;

	// backwards path
	for(int i = 0; i < maxIterations; ++i)
	{
	    unsigned short data[2];
	    // Read from input surface
	    surf2Dread(((uchar4*)data), velSurface, ((int)position.x) * 4, ((int)position.y));
	    float2 velocity;
	    velocity.x = __half2float(data[0]);
	    velocity.y = __half2float(data[1]);

	    // find delta s
	    float top,bottom,left,right;
	    if(velocity.y == 0.0f)
	    {
		top = bottom = FLT_MAX;
	    }
	    else
	    {
		top = (ceilf(position.y) - position.y) / -velocity.y;
		bottom = (floorf(position.y) - position.y) / -velocity.y;
	    }

	    if(velocity.x == 0.0f)
	    {
		left = right = FLT_MAX;
	    }
	    else
	    {
		right = (ceilf(position.x) - position.x) / -velocity.x;
		left = (floorf(position.x) - position.x) / -velocity.x;
	    }
	    
	    float deltas = FLT_MAX;
	    
	    if(top > 0.0f && top < deltas)
	    {
		deltas = top;
	    }
	    if(bottom > 0.0f && bottom < deltas)
	    {
		deltas = bottom;
	    }
	    if(right > 0.0f && right < deltas)
	    {
		deltas = right;
	    }
	    if(left > 0.0f && left < deltas)
	    {
		deltas = left;
	    }

	    // zero velocity, going nowhere
	    if(deltas == FLT_MAX)
	    {
		break;
	    }

	    // make sure we hit the next cell
	    deltas = deltas + ROUNDING_ADDITION;

	    // add weight/value

	    // end of line
	    if(weightTotal.y + deltas > length)
	    {
		break;
	    }

	    weightTotal.y = weightTotal.y + deltas;

	    // add weighted noise
	    pixelVal = pixelVal + (deltas * tex2D(noiseTex,(int)position.x,(int)position.y));

	    // find next point
	    position.x = position.x + deltas * -velocity.x;
	    position.y = position.y + deltas * -velocity.y;

	    // out of grid
	    if(position.x < 0.0f || position.x >= width || position.y < 0.0f || position.y >= height)
	    {
		break;
	    }
	}

	if((weightTotal.x + weightTotal.y) > 0.0f)
	{
	    pixelVal = pixelVal / (weightTotal.x + weightTotal.y);
	}

	// set pixel value
	unsigned short data[2];
	data[0] = __float2half_rn(pixelVal);
	data[1] = __float2half_rn(pixelDeb);
        surf2Dwrite(*((uchar4*)data), outSurface, x * 4, y);
    }
}

__global__ void makeTetListKernel(unsigned int * tetList, unsigned int * numTets, int totalTets, uint4 * ind, float3 * verts)
{
    int tetid = blockIdx.x*blockDim.x + threadIdx.x + (gridDim.x*blockDim.x*blockIdx.y);
    if(tetid >= totalTets)
    {
	return;
    }

    float4 tetpoints[4];
    *((float3*)&tetpoints[0]) = verts[ind[tetid].x];
    *((float3*)&tetpoints[1]) = verts[ind[tetid].y];
    *((float3*)&tetpoints[2]) = verts[ind[tetid].z];
    *((float3*)&tetpoints[3]) = verts[ind[tetid].w];

    // find viewing plane distance
    tetpoints[0].w = (tetpoints[0].x - planePoint.x) * planeNormal.x + (tetpoints[0].y - planePoint.y) * planeNormal.y + (tetpoints[0].z - planePoint.z) * planeNormal.z;
    tetpoints[1].w = (tetpoints[1].x - planePoint.x) * planeNormal.x + (tetpoints[1].y - planePoint.y) * planeNormal.y + (tetpoints[1].z - planePoint.z) * planeNormal.z;
    tetpoints[2].w = (tetpoints[2].x - planePoint.x) * planeNormal.x + (tetpoints[2].y - planePoint.y) * planeNormal.y + (tetpoints[2].z - planePoint.z) * planeNormal.z;
    tetpoints[3].w = (tetpoints[3].x - planePoint.x) * planeNormal.x + (tetpoints[3].y - planePoint.y) * planeNormal.y + (tetpoints[3].z - planePoint.z) * planeNormal.z;

    // determine if plane passes through tet
    int count = 0;
    if(tetpoints[0].w > 0.0f)
    {
	count++;
    }
    if(tetpoints[1].w > 0.0f)
    {
	count++;
    }
    if(tetpoints[2].w > 0.0f)
    {
	count++;
    }
    if(tetpoints[3].w > 0.0f)
    {
	count++;
    }

    if(count == 0 || count == 4)
    {
	return;
    }

    // project points onto plane and find basis values
    float3 projpoint;
    float2 basisMin;
    float2 basisMax;
    float tempx;
    projpoint.x = tetpoints[0].x - (planeNormal.x * tetpoints[0].w) - planePoint.x;
    projpoint.y = tetpoints[0].y - (planeNormal.y * tetpoints[0].w) - planePoint.y;
    projpoint.z = tetpoints[0].z - (planeNormal.z * tetpoints[0].w) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = basisMax.x = tempx;
    basisMin.y = basisMax.y = projpoint.y;
    
    projpoint.x = tetpoints[1].x - (planeNormal.x * tetpoints[1].w) - planePoint.x;
    projpoint.y = tetpoints[1].y - (planeNormal.y * tetpoints[1].w) - planePoint.y;
    projpoint.z = tetpoints[1].z - (planeNormal.z * tetpoints[1].w) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    projpoint.x = tetpoints[2].x - (planeNormal.x * tetpoints[2].w) - planePoint.x;
    projpoint.y = tetpoints[2].y - (planeNormal.y * tetpoints[2].w) - planePoint.y;
    projpoint.z = tetpoints[2].z - (planeNormal.z * tetpoints[2].w) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    projpoint.x = tetpoints[3].x - (planeNormal.x * tetpoints[3].w) - planePoint.x;
    projpoint.y = tetpoints[3].y - (planeNormal.y * tetpoints[3].w) - planePoint.y;
    projpoint.z = tetpoints[3].z - (planeNormal.z * tetpoints[3].w) - planePoint.z;

    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    /*basisMin.x = ceilf(basisMin.x - 0.5f) + 0.5f;
    basisMin.y = ceilf(basisMin.y - 0.5f) + 0.5f;
    basisMax.x = floorf(basisMax.x - 0.5f) + 0.500001f;
    basisMax.y = floorf(basisMax.y - 0.5f) + 0.500001f;*/

    basisMin.x = floorf(basisMin.x - 0.5f) - 0.5f;
    basisMin.y = floorf(basisMin.y - 0.5f) - 0.5f;
    basisMax.x = ceilf(basisMax.x + 0.5f) + 0.500001f;
    basisMax.y = ceilf(basisMax.y + 0.5f) + 0.500001f;

    if(basisMin.x > basisMax.x || basisMin.y > basisMax.y)
    {
	// no critical points in tet
	return;
    }

    //find if tet is outside the basis bounds
    if(basisMin.x > texXMax || basisMax.x < texXMin || basisMin.y > texYMax || basisMax.y < texYMin)
    {
	return;
    }

    unsigned int index = atomicAdd(numTets,(unsigned int)1);
    tetList[index] = tetid;
}
