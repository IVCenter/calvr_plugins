#ifndef PARTICLE_DREAMS_CUDA_H
#define PARTICLE_DREAMS_CUDA_H

#define DEBUG 0
#define REFL_HITS 1


// note injt and refl: inj 0 is gloobel only used for nun of inj, row 0 is not used
// therefore there is extivily 1 less row and 1 les injector refl that the number indicates.
#define CUDA_MESH_WIDTH 2048
#define CUDA_MESH_HEIGHT 1024
#define TARGET_FR_RATE 30
// only 256 512 1024 2048 have been tested

//#define CUDA_MESH_WIDTH 256
//#define CUDA_MESH_HEIGHT 256
#define PDATA_ROW_SIZE 7

#define REFL_DATA_MUNB 20
#define REFL_DATA_ROWS 8
#define REFL_DATA_ROW_ELEM 3
#define INJT_DATA_MUNB 20
#define INJT_DATA_ROWS 8
#define INJT_DATA_ROW_ELEM 3
#define ENABLE_SOUND_SERV 1
#define ENABLE_SOUND_POS_UPDATES 1
#define SHOW_MARKERS 0

#define DEBUG_PRINT 0
#define FR_RATE_PRINT 0

#include <cuda.h>
#include <cuda_runtime.h>

void setReflData(void * data, int size);
void setInjData(void * data, int size);

void launchPoint1(float3* pos, float4* color, float * pdata,float * debugData ,unsigned int width,
    unsigned int height, int max_age,int disappear_age,float alphaControl, float time, float gravity, float colorFreq, float r3);

__global__ void Point1(float3* pos, float4* color, float * pdata,float * debugData ,unsigned int width,
	unsigned int height, int max_age,int disappear_age,float alphaControl, float time, float gravity, float colorFreq, float r3);

__global__ void PointSquars(float4* pos, float * pdata, unsigned int width,
	unsigned int height, int max_age, float time, float r1, float r2, float r3);

#endif
