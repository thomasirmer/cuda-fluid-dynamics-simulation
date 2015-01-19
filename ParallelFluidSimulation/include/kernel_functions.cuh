
#ifndef _KERNEL_FUNCTIONS_CUH_
#define _KERNEL_FUNCTIONS_CUH_

__device__ int getGlobalThreadId();

__device__ float getVectorLength(float xCoord, float yCoord);
__device__ float getVectorAngle(float xCoord, float yCoord);

__global__ void simulate(float *outputValues, const float *inputValues);

#endif // _KERNEL_FUNCTIONS_CUH_
