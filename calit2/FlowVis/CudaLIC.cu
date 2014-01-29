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
__constant__ float invBasisMat[9];
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

void setPlaneConsts(void * point, void * normal, void * right, void * up, void * rightNorm, void * upNorm, void * basisMat, void * bLength)
{
    cudaMemcpyToSymbol(planePoint,point,sizeof(float3));
    cudaMemcpyToSymbol(planeNormal,normal,sizeof(float3));
    cudaMemcpyToSymbol(planeRight,right,sizeof(float3));
    cudaMemcpyToSymbol(planeUp,up,sizeof(float3));
    cudaMemcpyToSymbol(planeRightNorm,rightNorm,sizeof(float3));
    cudaMemcpyToSymbol(planeUpNorm,upNorm,sizeof(float3));
    cudaMemcpyToSymbol(invBasisMat,basisMat,9*sizeof(float));
    cudaMemcpyToSymbol(basisLength,bLength,sizeof(float));

    /*float devptr[9];

    // verify
    float * hostptr = (float*)rightNorm;
    cudaMemcpyFromSymbol(devptr,planeRightNorm,sizeof(float3));
    std::cerr << "Host rightpointN x: " << hostptr[0] << " y: " << hostptr[1] << " z: " << hostptr[2] << std::endl;
    std::cerr << "Dev rightpointN x: " << devptr[0] << " y: " << devptr[1] << " z: " << devptr[2] << std::endl;*/

}

void setTexConsts(void * xMin, void * xMax, void * yMin, void * yMax)
{
    cudaMemcpyToSymbol(texXMin,xMin,sizeof(float));
    cudaMemcpyToSymbol(texXMax,xMax,sizeof(float));
    cudaMemcpyToSymbol(texYMin,yMin,sizeof(float));
    cudaMemcpyToSymbol(texYMax,yMax,sizeof(float));

    std::cerr << "Setting basis range X: " << ((float*)xMin)[0] << " " << ((float*)xMax)[0] << " Y: " << ((float*)yMin)[0] << " " << ((float*)yMax)[0] << std::endl;
}

void launchVel(uint4 * indices, float3 * verts, float3 * velocity, unsigned int * tetList, int numTets, int width, int height)
{
    //std::cerr << "Running velocity kernel" << std::endl;
    /*float devptr[9];
    cudaMemcpyFromSymbol(devptr,planePoint,sizeof(float3));
    std::cerr << "Dev planepoint x: " << devptr[0] << " y: " << devptr[1] << " z: " << devptr[2] << std::endl;*/

    dim3 blockc(8,8,1);
    dim3 gridc(width/8,height/8,1);

    velTest<<< gridc, blockc >>>(width,height);
    cudaThreadSynchronize();

    dim3 block(256,1,1);
    int griddim = (numTets / 256) + 1;
    //std::cerr << "NumTets: " << numTets << " Grid dim: " << griddim << std::endl;
    dim3 grid(griddim,1,1);

    velKernel<<< grid, block >>>(indices,verts,velocity,tetList,numTets,width,height,width/2.0,height/2.0);
}

void launchLIC(int width, int height, float length, cudaArray * noiseArray)
{
    //std::cerr << "Running LIC kernel" << std::endl;
    cudaBindTextureToArray(noiseTex,noiseArray);

    dim3 blockc(8,8,1);
    dim3 gridc(width/8,height/8,1);
    //dim3 gridc(1,1,1);

    licKernel<<< gridc, blockc >>>(width,height,length);
    //licKernel<<< gridc, blockc >>>(1024,1024,length);
    cudaThreadSynchronize();

    cudaUnbindTexture(noiseTex);
}

void launchMakeTetList(unsigned int * tetList, unsigned int * numTets, int totalTets, uint4 * indices, float3 * verts)
{
    dim3 block(256,1,1);
    int griddim = (totalTets / 256) + 1;
    //std::cerr << "NumTets: " << numTets << " Grid dim: " << griddim << std::endl;
    dim3 grid(griddim,1,1);

    makeTetListKernel<<< grid, block >>>(tetList,numTets,totalTets,indices,verts);
}

__global__ void velTest(int width, int height) 
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        unsigned short data[2];
	data[0] = __float2half_rn(0.0);
	data[1] = __float2half_rn(0.0);
        // Read from input surface
        //surf2Dread(&data,  inputSurfRef, x * 4, y);
        // Write to output surface
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

    /*if(tetid == 0)
    {
	printf("PlaneUp %f %f %f\n",planeUp.x,planeUp.y,planeUp.z);
	printf("PlaneUpN %f %f %f\n",planeUpNorm.x,planeUpNorm.y,planeUpNorm.z);
	printf("PlaneRight %f %f %f\n",planeRight.x,planeRight.y,planeRight.z);
	printf("PlaneRightN %f %f %f\n",planeRightNorm.x,planeRightNorm.y,planeRightNorm.z);
	printf("PlanePoint %f %f %f\n",planePoint.x,planePoint.y,planePoint.z);
	printf("PlaneNormal %f %f %f\n",planeNormal.x,planeNormal.y,planeNormal.z);
	printf("BasisMat: %f %f %f %f %f %f %f %f %f\n",invBasisMat[0],invBasisMat[1],invBasisMat[2],invBasisMat[3],invBasisMat[4],invBasisMat[5],invBasisMat[6],invBasisMat[7],invBasisMat[8]);
    }*/

    float4 tetpoints[4];
    *((float3*)&tetpoints[0]) = verts[ind[tetid].x];
    *((float3*)&tetpoints[1]) = verts[ind[tetid].y];
    *((float3*)&tetpoints[2]) = verts[ind[tetid].z];
    *((float3*)&tetpoints[3]) = verts[ind[tetid].w];

    //printf("Point0 %f %f %f\n",tetpoints[0].x,tetpoints[0].y,tetpoints[0].z);
    //printf("Point1 %f %f %f\n",tetpoints[1].x,tetpoints[1].y,tetpoints[1].z);
    //printf("Point2 %f %f %f\n",tetpoints[2].x,tetpoints[2].y,tetpoints[2].z);
    //printf("Point3 %f %f %f\n",tetpoints[3].x,tetpoints[3].y,tetpoints[3].z);

    // find viewing plane distance
    tetpoints[0].w = (tetpoints[0].x - planePoint.x) * planeNormal.x + (tetpoints[0].y - planePoint.y) * planeNormal.y + (tetpoints[0].z - planePoint.z) * planeNormal.z;
    tetpoints[1].w = (tetpoints[1].x - planePoint.x) * planeNormal.x + (tetpoints[1].y - planePoint.y) * planeNormal.y + (tetpoints[1].z - planePoint.z) * planeNormal.z;
    tetpoints[2].w = (tetpoints[2].x - planePoint.x) * planeNormal.x + (tetpoints[2].y - planePoint.y) * planeNormal.y + (tetpoints[2].z - planePoint.z) * planeNormal.z;
    tetpoints[3].w = (tetpoints[3].x - planePoint.x) * planeNormal.x + (tetpoints[3].y - planePoint.y) * planeNormal.y + (tetpoints[3].z - planePoint.z) * planeNormal.z;

    //printf("Distances: %f %f %f %f\n",tetpoints[0].w,tetpoints[1].w,tetpoints[2].w,tetpoints[3].w);

    // determine if plane passes through tet
    /*int count = 0;
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
    }*/

    // project points onto plane and find basis values
    float3 projpoint;
    float2 basisMin;
    float2 basisMax;
    float tempx;
    projpoint.x = tetpoints[0].x - (planeNormal.x * tetpoints[0].w) - planePoint.x;
    projpoint.y = tetpoints[0].y - (planeNormal.y * tetpoints[0].w) - planePoint.y;
    projpoint.z = tetpoints[0].z - (planeNormal.z * tetpoints[0].w) - planePoint.z;

    //printf("projpoint: %f %f %f\n",projpoint.x,projpoint.y,projpoint.z);

    //tempx = projpoint.x * invBasisMat[0] + projpoint.y * invBasisMat[3] + projpoint.z * invBasisMat[6];
    //projpoint.y = projpoint.x * invBasisMat[1] + projpoint.y * invBasisMat[4] + projpoint.z * invBasisMat[7];
    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    //printf("basis %f %f\n",tempx,projpoint.y);

    basisMin.x = basisMax.x = tempx;
    basisMin.y = basisMax.y = projpoint.y;
    
    projpoint.x = tetpoints[1].x - (planeNormal.x * tetpoints[1].w) - planePoint.x;
    projpoint.y = tetpoints[1].y - (planeNormal.y * tetpoints[1].w) - planePoint.y;
    projpoint.z = tetpoints[1].z - (planeNormal.z * tetpoints[1].w) - planePoint.z;

    //tempx = projpoint.x * invBasisMat[0] + projpoint.y * invBasisMat[3] + projpoint.z * invBasisMat[6];
    //projpoint.y = projpoint.x * invBasisMat[1] + projpoint.y * invBasisMat[4] + projpoint.z * invBasisMat[7];
    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;


    //printf("basis %f %f\n",tempx,projpoint.y);

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    projpoint.x = tetpoints[2].x - (planeNormal.x * tetpoints[2].w) - planePoint.x;
    projpoint.y = tetpoints[2].y - (planeNormal.y * tetpoints[2].w) - planePoint.y;
    projpoint.z = tetpoints[2].z - (planeNormal.z * tetpoints[2].w) - planePoint.z;

    //tempx = projpoint.x * invBasisMat[0] + projpoint.y * invBasisMat[3] + projpoint.z * invBasisMat[6];
    //projpoint.y = projpoint.x * invBasisMat[1] + projpoint.y * invBasisMat[4] + projpoint.z * invBasisMat[7];
    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    //printf("basis %f %f\n",tempx,projpoint.y);

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    projpoint.x = tetpoints[3].x - (planeNormal.x * tetpoints[3].w) - planePoint.x;
    projpoint.y = tetpoints[3].y - (planeNormal.y * tetpoints[3].w) - planePoint.y;
    projpoint.z = tetpoints[3].z - (planeNormal.z * tetpoints[3].w) - planePoint.z;

    //tempx = projpoint.x * invBasisMat[0] + projpoint.y * invBasisMat[3] + projpoint.z * invBasisMat[6];
    //projpoint.y = projpoint.x * invBasisMat[1] + projpoint.y * invBasisMat[4] + projpoint.z * invBasisMat[7];
    tempx = (projpoint.x * planeUpNorm.x + projpoint.y * planeUpNorm.y + projpoint.z * planeUpNorm.z) / basisLength;
    projpoint.y = (projpoint.x * planeRightNorm.x + projpoint.y * planeRightNorm.y + projpoint.z * planeRightNorm.z) / basisLength;

    //printf("basis %f %f\n",tempx,projpoint.y);

    basisMin.x = fminf(basisMin.x,tempx);
    basisMax.x = fmaxf(basisMax.x,tempx);
    basisMin.y = fminf(basisMin.y,projpoint.y);
    basisMax.y = fmaxf(basisMax.y,projpoint.y);

    //printf("X: %f %f\n",basisMin.x,basisMax.x);
    //printf("Y: %f %f\n",basisMin.y,basisMax.y);

    /*basisMin.x = ceilf(basisMin.x - 0.5f) + 0.5f;
    basisMin.y = ceilf(basisMin.y - 0.5f) + 0.5f;
    basisMax.x = floorf(basisMax.x - 0.5f) + 0.500001f;
    basisMax.y = floorf(basisMax.y - 0.5f) + 0.500001f;*/

    basisMin.x = floorf(basisMin.x - 0.5f) - 0.5f;
    basisMin.y = floorf(basisMin.y - 0.5f) - 0.5f;
    basisMax.x = ceilf(basisMax.x + 0.5f) + 0.500001f;
    basisMax.y = ceilf(basisMax.y + 0.5f) + 0.500001f;

    //printf("X: %f %f\n",basisMin.x,basisMax.x);
    //printf("Y: %f %f\n",basisMin.y,basisMax.y);

    if(basisMin.x > basisMax.x || basisMin.y > basisMax.y)
    {
	//printf("Basis range exit x: %f %f y: %f %f\n",basisMin.x,basisMax.x,basisMin.y,basisMax.y);
	// no critical points in tet
	return;
    }

    //find if tet is outside the basis bounds
    if(basisMin.x > texXMax || basisMax.x < texXMin || basisMin.y > texYMax || basisMax.y < texYMin)
    {
	//printf("X: %f %f\n",basisMin.x,basisMax.x);
	//printf("Y: %f %f\n",basisMin.y,basisMax.y);
	return;
    }

    /*if(tetid < 1024*1024)
    {
	unsigned short data[2];
	data[0] = __float2half_rn(0.5);
	data[1] = __float2half_rn(0.5);
	surf2Dwrite(*((uchar4*)data), velSurface, (tetid % 1024) * 4, tetid / 1024);
    }*/

    /*if(basisMin.x >= -512 && basisMax.x <= 512 && basisMin.y >= -512 && basisMax.y <= 512)
    {
	if(tetid < 1024*1024)
		{
		    unsigned short data[2];
		    data[0] = __float2half_rn(1.0);
		    data[1] = __float2half_rn(0.0);
		    surf2Dwrite(*((uchar4*)data), velSurface, (tetid % 1024) * 4, tetid / 1024);
		}
    }
    else
    {
	if(tetid < 1024*1024)
		{
		    unsigned short data[2];
		    data[0] = __float2half_rn(0.5);
		    data[1] = __float2half_rn(0.5);
		    surf2Dwrite(*((uchar4*)data), velSurface, (tetid % 1024) * 4, tetid / 1024);
		}
    }*/

    //return;

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

    //printf("barMat:\n%f %f %f\n%f %f %f\n%f %f %f\n",tetMat[0],tetMat[1],tetMat[2],tetMat[3],tetMat[4],tetMat[5],tetMat[6],tetMat[7],tetMat[8]);

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

	    //printf("temppoint %f %f %f\n",tempPoint.x,tempPoint.y,tempPoint.z);

	    float4 coords;
	    coords.x = tempPoint.x * tetMat[0] + tempPoint.y * tetMat[1] + tempPoint.z * tetMat[2];
	    coords.y = tempPoint.x * tetMat[3] + tempPoint.y * tetMat[4] + tempPoint.z * tetMat[5];
	    coords.z = tempPoint.x * tetMat[6] + tempPoint.y * tetMat[7] + tempPoint.z * tetMat[8];
	    coords.w = 1.0f - coords.x - coords.y - coords.z;

	    //printf("Coords %f %f %f %f\n",coords.x,coords.y,coords.z,coords.w);

	    if(coords.x > 1.0f + VEL_EPS || coords.x < 0.0f - VEL_EPS || coords.y > 1.0f + VEL_EPS || coords.y < 0.0f - VEL_EPS || coords.z > 1.0f + VEL_EPS || coords.z < 0.0f - VEL_EPS || coords.w > 1.0f + VEL_EPS || coords.w < 0.0f - VEL_EPS)
	    {	
		continue;
	    }

	    /*if(tetid < 1024*1024)
	    {
		unsigned short data[2];
		data[0] = __float2half_rn(0.0);
		data[1] = __float2half_rn(1.0);
		surf2Dwrite(*((uchar4*)data), velSurface, (tetid % 1024) * 4, tetid / 1024);
	    }*/

	    int2 texelIndex;
	    texelIndex.x = lrintf((floorf(j) + hwidth));
	    texelIndex.y = lrintf(floor(i) + hheight);
	    //printf("Texel index: %f %f %d %d\n",j,i,texelIndex.x,texelIndex.y);
	    if(texelIndex.x >= 0 && texelIndex.x < width && texelIndex.y >= 0 && texelIndex.y < height)
	    {
		/*if(tetid < 1024*1024)
		{
		    unsigned short data[2];
		    data[0] = __float2half_rn(0.0);
		    data[1] = __float2half_rn(1.0);
		    surf2Dwrite(*((uchar4*)data), velSurface, (tetid % 1024) * 4, tetid / 1024);
		}*/

		// find point velocity
		float3 myVelocity;
		myVelocity.x = coords.x * tetVel[0].x + coords.y * tetVel[1].x + coords.z * tetVel[2].x + coords.w * tetVel[3].x;
		myVelocity.y = coords.x * tetVel[0].y + coords.y * tetVel[1].y + coords.z * tetVel[2].y + coords.w * tetVel[3].y;
		myVelocity.z = coords.x * tetVel[0].z + coords.y * tetVel[1].z + coords.z * tetVel[2].z + coords.w * tetVel[3].z;

		// normalize
		float mag = myVelocity.x * myVelocity.x + myVelocity.y * myVelocity.y + myVelocity.z * myVelocity.z;
		mag = sqrt(mag);
		if(mag > 0.0f)
		{
		    myVelocity.x = myVelocity.x / mag;
		    myVelocity.y = myVelocity.y / mag;
		    myVelocity.z = myVelocity.z / mag;
		}

		unsigned short data[2];
		// project onto texture basis
		/*float2 output;
		output.x = planeRightNorm.x * myVelocity.x + planeRightNorm.y * myVelocity.y + planeRightNorm.z * myVelocity.z;
		output.y = planeUpNorm.x * myVelocity.x + planeUpNorm.y * myVelocity.y + planeUpNorm.z * myVelocity.z;
		printf("Output x: %d y: %d %f %f\n",texelIndex.x,texelIndex.y,output.x,output.y);
		data[0] = __float2half_rn(output.x);
		data[1] = __float2half_rn(output.y);*/
		data[0] = __float2half_rn(planeRightNorm.x * myVelocity.x + planeRightNorm.y * myVelocity.y + planeRightNorm.z * myVelocity.z);
		data[1] = __float2half_rn(planeUpNorm.x * myVelocity.x + planeUpNorm.y * myVelocity.y + planeUpNorm.z * myVelocity.z);
		/*if(output.x != 0.0f || output.y != 0.0f)
		{
		    data[0] = __float2half_rn(1.0f);
		    data[1] = __float2half_rn(0.0f);
		}
		else
		{
		    data[0] = __float2half_rn(0.0f);
		    data[1] = __float2half_rn(1.0f);
		    printf("Coords %f %f %f %f\n",coords.x,coords.y,coords.z,coords.w);
		    printf("Vel %f %f %f\n",tetVel[0].x,tetVel[0].y,tetVel[0].z);
		    printf("Vel %f %f %f\n",tetVel[1].x,tetVel[1].y,tetVel[1].z);
		    printf("Vel %f %f %f\n",tetVel[2].x,tetVel[2].y,tetVel[2].z);
		    printf("Vel %f %f %f\n",tetVel[3].x,tetVel[3].y,tetVel[3].z);
		}*/
		surf2Dwrite(*((uchar4*)data), velSurface, texelIndex.x * 4, texelIndex.y);
	    }
	}
    }
}

#define ROUNDING_ADDITION 0.0001f

__global__ void licKernel(int width, int height, float length)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    //unsigned int x = 512;
    //unsigned int y = 512;

    //printf("X: %d Y: %d w: %d h: %d\n",x,y,width,height);

    //if (x < width && y < height) {
	/*unsigned short data[2];
	//data[0] = __float2half_rn(tex2D(noiseTex,x,y));
	data[0] = __float2half_rn(0.0f);
	data[1] = __float2half_rn(1.0f);
	surf2Dwrite(*((uchar4*)data), outSurface, x * 4, y);*/

	/*unsigned short data[2];
	// Read from input surface
	surf2Dread(((uchar4*)data), velSurface, x * 4, y);
	float2 velocity;
	velocity.x = __half2float(data[0]);
	velocity.y = __half2float(data[1]);

	printf("Coord x: %d y: %d Velocity: %f %f\n",x,y,velocity.x,velocity.y);*/
    //}

    float pixelVal = 0.0f;
    //float pixelDeb = 0.0f;
    // one thread per pixel
    if (x < width && y < height)
    {
	float2 position;
	float2 weightTotal;

	// init to pixel center
	position.x = x + 0.5f;
	position.y = y + 0.5f;

	weightTotal.x = weightTotal.y = 0.0f;

	int maxIterations = length * 10.0f;
	// forward path
	for(int i = 0; i < maxIterations; ++i)
	{
	    //printf("Position: %f %f\n",position.x,position.y);

	    unsigned short data[2];
	    // Read from input surface
	    surf2Dread(((uchar4*)data), velSurface, ((int)position.x) * 4, ((int)position.y));
	    float2 velocity;
	    velocity.x = __half2float(data[0]);
	    velocity.y = __half2float(data[1]);

	    //printf("Velocity: %f %f\n",velocity.x,velocity.y);

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
	    
	    //printf("T: %f B: %f R: %f L: %f\n",top,bottom,right,left);

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

	    //printf("DeltaS: %f\n",deltas);

	    // zero velocity, going nowhere
	    if(deltas == FLT_MAX)
	    {
		//printf("FLT_MAX break\n");
		break;
	    }

	    // make sure we hit the next cell
	    deltas = deltas + ROUNDING_ADDITION;

	    // add weight/value

	    // end of line
	    if(weightTotal.x + deltas > length)
	    {
		//printf("weightTotal break\n");
		break;
	    }

	    weightTotal.x = weightTotal.x + deltas;

	    //printf("WeightTotal: %f\n",weightTotal.x);

	    // add weighted noise
	    pixelVal = pixelVal + (deltas * tex2D(noiseTex,(int)position.x,(int)position.y));
	    //printf("PixelVal: %f\n",pixelVal);

	    // find next point
	    position.x = position.x + deltas * velocity.x;
	    position.y = position.y + deltas * velocity.y;

	    // out of grid
	    if(position.x < 0.0f || position.x >= width || position.y < 0.0f || position.y >= height)
	    {
		//printf("Out of grid break\n");
		break;
	    }
	}

	position.x = x + 0.5f;
	position.y = y + 0.5f;

	// backwards path
	for(int i = 0; i < maxIterations; ++i)
	{
	    //printf("Position: %f %f\n",position.x,position.y);

	    unsigned short data[2];
	    // Read from input surface
	    surf2Dread(((uchar4*)data), velSurface, ((int)position.x) * 4, ((int)position.y));
	    float2 velocity;
	    velocity.x = __half2float(data[0]);
	    velocity.y = __half2float(data[1]);

	    //printf("Velocity: %f %f\n",velocity.x,velocity.y);

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
	    
	    //printf("T: %f B: %f R: %f L: %f\n",top,bottom,right,left);

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

	    //printf("DeltaS: %f\n",deltas);

	    // zero velocity, going nowhere
	    if(deltas == FLT_MAX)
	    {
		//printf("FLT_MAX break\n");
		break;
	    }

	    // make sure we hit the next cell
	    deltas = deltas + ROUNDING_ADDITION;

	    // add weight/value

	    // end of line
	    if(weightTotal.y + deltas > length)
	    {
		//printf("weightTotal break\n");
		break;
	    }

	    weightTotal.y = weightTotal.y + deltas;

	    //printf("WeightTotal: %f\n",weightTotal.x);

	    // add weighted noise
	    pixelVal = pixelVal + (deltas * tex2D(noiseTex,(int)position.x,(int)position.y));
	    //printf("PixelVal: %f\n",pixelVal);

	    // find next point
	    position.x = position.x + deltas * -velocity.x;
	    position.y = position.y + deltas * -velocity.y;

	    // out of grid
	    if(position.x < 0.0f || position.x >= width || position.y < 0.0f || position.y >= height)
	    {
		//printf("Out of grid break\n");
		break;
	    }
	}

	if((weightTotal.x + weightTotal.y) > 0.0f)
	{
	    pixelVal = pixelVal / (weightTotal.x + weightTotal.y);
	}
	else
	{
	    //pixelDeb = 1.0f;
	}

	//printf("Writing pixelVal: %f\n",pixelVal);

	// set pixel value
	unsigned short data[2];
	data[0] = __float2half_rn(pixelVal);
	data[1] = __float2half_rn(0.0);
        surf2Dwrite(*((uchar4*)data), outSurface, x * 4, y);
    }
}

__global__ void makeTetListKernel(unsigned int * tetList, unsigned int * numTets, int totalTets, uint4 * ind, float3 * verts)
{
    int tetid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tetid >= totalTets)
    {
	return;
    }

    /*if(tetid == 0)
    {
	printf("PlaneUp %f %f %f\n",planeUp.x,planeUp.y,planeUp.z);
	printf("PlaneUpN %f %f %f\n",planeUpNorm.x,planeUpNorm.y,planeUpNorm.z);
	printf("PlaneRight %f %f %f\n",planeRight.x,planeRight.y,planeRight.z);
	printf("PlaneRightN %f %f %f\n",planeRightNorm.x,planeRightNorm.y,planeRightNorm.z);
	printf("PlanePoint %f %f %f\n",planePoint.x,planePoint.y,planePoint.z);
	printf("PlaneNormal %f %f %f\n",planeNormal.x,planeNormal.y,planeNormal.z);
	printf("BasisMat: %f %f %f %f %f %f %f %f %f\n",invBasisMat[0],invBasisMat[1],invBasisMat[2],invBasisMat[3],invBasisMat[4],invBasisMat[5],invBasisMat[6],invBasisMat[7],invBasisMat[8]);
    }*/

    float4 tetpoints[4];
    *((float3*)&tetpoints[0]) = verts[ind[tetid].x];
    *((float3*)&tetpoints[1]) = verts[ind[tetid].y];
    *((float3*)&tetpoints[2]) = verts[ind[tetid].z];
    *((float3*)&tetpoints[3]) = verts[ind[tetid].w];

    //printf("Point0 %f %f %f\n",tetpoints[0].x,tetpoints[0].y,tetpoints[0].z);
    //printf("Point1 %f %f %f\n",tetpoints[1].x,tetpoints[1].y,tetpoints[1].z);
    //printf("Point2 %f %f %f\n",tetpoints[2].x,tetpoints[2].y,tetpoints[2].z);
    //printf("Point3 %f %f %f\n",tetpoints[3].x,tetpoints[3].y,tetpoints[3].z);

    // find viewing plane distance
    tetpoints[0].w = (tetpoints[0].x - planePoint.x) * planeNormal.x + (tetpoints[0].y - planePoint.y) * planeNormal.y + (tetpoints[0].z - planePoint.z) * planeNormal.z;
    tetpoints[1].w = (tetpoints[1].x - planePoint.x) * planeNormal.x + (tetpoints[1].y - planePoint.y) * planeNormal.y + (tetpoints[1].z - planePoint.z) * planeNormal.z;
    tetpoints[2].w = (tetpoints[2].x - planePoint.x) * planeNormal.x + (tetpoints[2].y - planePoint.y) * planeNormal.y + (tetpoints[2].z - planePoint.z) * planeNormal.z;
    tetpoints[3].w = (tetpoints[3].x - planePoint.x) * planeNormal.x + (tetpoints[3].y - planePoint.y) * planeNormal.y + (tetpoints[3].z - planePoint.z) * planeNormal.z;

    //printf("Distances: %f %f %f %f\n",tetpoints[0].w,tetpoints[1].w,tetpoints[2].w,tetpoints[3].w);

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

    unsigned int index = atomicAdd(numTets,(unsigned int)1);
    tetList[index] = tetid;
}
