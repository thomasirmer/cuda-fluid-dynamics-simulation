#ifndef _KERNEL_FUNCTIONS_CUH_
#define _KERNEL_FUNCTIONS_CUH_

#include <defines.cuh>

__device__ int getGlobalThreadId();

__device__ float getVectorLength(float xCoord, float yCoord);
__device__ float getVectorAngle(float xCoord, float yCoord);

__device__ void hsv2rgb(unsigned int hue, unsigned int sat, unsigned int val,
		unsigned char * r, unsigned char * g, unsigned char * b,
		unsigned char maxBrightness);

__global__ void simulate(Vector* inDots, Vector* outDots);
#endif // _KERNEL_FUNCTIONS_CUH_
