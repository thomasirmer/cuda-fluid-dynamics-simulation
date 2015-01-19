#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cpu_anim.h"
#include "utils.h"
#include "defines.cuh"

#define SIM_WIDTH  1280
#define SIM_HEIGHT 720

//#define _DEBUG_

// KERNEL FUNCTIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// global threadID for 2D grid of 2D blocks
__device__ int getGlobalThreadId() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;

	return threadId;
}

// vector calculations
__device__ float getVectorLength(float xCoord, float yCoord) {
	return (float) sqrt(xCoord * xCoord + yCoord * yCoord);
}

// return the angle of the vector
__device__ float getVectorAngle(float xCoord, float yCoord) {
	float angleRad = atan2(yCoord, xCoord);
	float angleDeg = (angleRad / M_PI) * 180.0f;
	return angleDeg;
}

// simulation function (copy input --> output)
__global__ void simulate(float *outputValues, const float *inputValues) {
	int threadId = getGlobalThreadId();

	if (threadId < SIM_WIDTH * SIM_HEIGHT) {
		outputValues[threadId * 2] = inputValues[threadId * 2];
		outputValues[threadId * 2 + 1] = inputValues[threadId * 2 + 1];
	}
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

void anim_exit(DataBlock *d) {
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);
}

void anim_gpu(DataBlock *d, int ticks) {
// TODO: Animation
	dim3 blocks(ceil(SIM_WIDTH / 32.0f), ceil(SIM_HEIGHT / 32.0f));
	dim3 threads(32, 32);
	CPUAnimBitmap* bitmap = d->bitmap;

//copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
	simulate<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
	swap(d->dev_inSrc, d->dev_outSrc);

// TODO: Implement float_to_color that it uses both values!
	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_outSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(),
			cudaMemcpyDeviceToHost);

#ifdef _DEBUG_
	unsigned char* h_bitmap = new unsigned char[bitmap->image_size() * 2];
	cudaMemcpy(h_bitmap, d->output_bitmap, bitmap->image_size() * 2, cudaMemcpyDeviceToHost);
	printf("%d\n", h_bitmap[0]);
#endif
}

int main(void) {
	printf("Starting CUDA-Application - Parallel Fluid Simulation ...\n");

// TODO: initialize all the stuff
//Vector2D* vectorField = new Vector2D[SIM_HEIGHT * SIM_WIDTH];
	Vector* vectorField = new Vector[SIM_HEIGHT * SIM_WIDTH];

// TODO: do simulation (ihno)

// TODO: visualize (thomas)
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	DataBlock dataBlock;
	CPUAnimBitmap animBitmap(SIM_WIDTH, SIM_HEIGHT, &dataBlock);
	dataBlock.bitmap = &animBitmap;

// allocate device memory
	int imageSize = animBitmap.image_size(); // image_size() returns width * height * 4
	cudaMalloc((void**) &dataBlock.output_bitmap, imageSize * 2);
	cudaMalloc((void**) &dataBlock.dev_inSrc, imageSize * 2);
	cudaMalloc((void**) &dataBlock.dev_outSrc, imageSize * 2);

// random initialize vectorField
	for (int i = 0; i < SIM_HEIGHT * SIM_WIDTH; i++) {
		// random values between [0.0 ... 1.0]
		//float xValue = (float) rand() / (float) RAND_MAX;
		//float yValue = (float) rand() / (float) RAND_MAX;

		// circle values
		float xValue = cos(((float) i / (SIM_WIDTH * SIM_HEIGHT - 1)) * 2 * M_PI);
		float yValue = sin(((float) i / (SIM_WIDTH * SIM_HEIGHT - 1)) * 2 * M_PI);

		// assign to vectorField
		vectorField[i][0] = xValue;
		vectorField[i][1] = yValue;
	}

// copy input values to device
	cudaMemcpy(dataBlock.dev_inSrc, vectorField, imageSize * 2,
			cudaMemcpyHostToDevice);

// start simulation
	animBitmap.anim_and_exit((void (*)(void*, int)) anim_gpu, (void (*)(void*))anim_exit );
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
}
