#ifndef _KERNEL_FUNCTIONS_CUH_
#define _KERNEL_FUNCTIONS_CUH_

// ****************************************************************************
// HELPER FUNCTIONS
// ****************************************************************************

__device__ int getGlobalThreadId();
__device__ int getArrayOffset();

// ****************************************************************************
// VECTOR CALCULATIONS
// ****************************************************************************

__device__ float getVectorLength(float xCoord, float yCoord);
__device__ float getVectorAngle(float xCoord, float yCoord);
__device__ float getAngleBetween(float x1, float y1, float x2, float y2);

// ****************************************************************************
// COLOR CONVERSION
// ****************************************************************************

__device__ void hsv2rgb(unsigned int hue, unsigned int sat, unsigned int val, unsigned char * r, unsigned char * g, unsigned char * b,
		unsigned char maxBrightness);

#endif // _KERNEL_FUNCTIONS_CUH_
