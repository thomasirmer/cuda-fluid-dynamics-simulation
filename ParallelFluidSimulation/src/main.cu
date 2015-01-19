#include <stdio.h>
#include <stdlib.h>

#include "cpu_anim.h"
#include "utils.h"
#include "defines.cuh"

#define SIM_WIDTH  128
#define SIM_HEIGHT 128

#define _DEBUG_

// 2D grid of 2D blocks
__device__ int getGlobalThreadId() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;

	return threadId;
}

__global__ void copy_const_kernel(float *inputValues, const float *constantValues) {
	int threadId = getGlobalThreadId();

	inputValues[threadId * 2] = constantValues[threadId * 2];
	inputValues[threadId * 2 + 1] = constantValues[threadId * 2 + 1];
}

__global__ void simulate(float *outputValues, const float *inputValues) {
	int threadId = getGlobalThreadId();

	outputValues[threadId * 2] = inputValues[threadId * 2];
	outputValues[threadId * 2 + 1] = inputValues[threadId * 2 + 1];
}

void anim_exit(DataBlock *d) {
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);
}

void anim_gpu(DataBlock *d, int ticks) {
	// TODO: Animation
	dim3 blocks(SIM_WIDTH / 32, SIM_HEIGHT / 32);
	dim3 threads(32, 32);
	CPUAnimBitmap* bitmap = d->bitmap;

	//copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
	simulate<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
	swap(d->dev_inSrc, d->dev_outSrc);

	// TODO: Implement this function that it uses both values!
	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_outSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

#ifdef _DEBUG_
	float* vectorField = new float[SIM_HEIGHT*SIM_WIDTH*2];
	cudaMemcpy(vectorField, d->dev_outSrc, SIM_HEIGHT*SIM_WIDTH*2*sizeof(float), cudaMemcpyDeviceToHost);
	printf("x1: %f, y1: %f, x2: %f, y2: %f\n", vectorField[0], vectorField[1], vectorField[2], vectorField[3]);
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
	cudaMalloc((void**) &dataBlock.dev_constSrc, imageSize * 2);

	// initialize constant data (border pixels are all zero)
	// top border
	for (int i = 0; i < SIM_WIDTH; i++) {
		vectorField[i][0] = 0;
		vectorField[i][1] = 0;
	}
	// bottom border
	for (int i = SIM_WIDTH * SIM_HEIGHT - SIM_WIDTH; i < SIM_WIDTH * SIM_HEIGHT;
			i++) {
		vectorField[i][0] = 0;
		vectorField[i][1] = 0;
	}
	// left border
	for (int i = 0; i < SIM_WIDTH * SIM_HEIGHT; i += SIM_WIDTH) {
		vectorField[i][0] = 0;
		vectorField[i][1] = 0;
	}
	// right border
	for (int i = SIM_WIDTH; i < SIM_WIDTH * SIM_HEIGHT; i += SIM_WIDTH) {
		vectorField[i][0] = 0;
		vectorField[i][1] = 0;
	}
	// set all other values to zero
	for (int i = 0; i < SIM_HEIGHT * SIM_WIDTH; i++) {
		vectorField[i][0] = 0.0;
		vectorField[i][1] = 0.0;
	}

	// copy constant values to device
	cudaMemcpy(dataBlock.dev_constSrc, vectorField, imageSize * 2, cudaMemcpyHostToDevice);

	// random initialize vectorField
	for (int i = 0; i < SIM_HEIGHT * SIM_WIDTH; i++) {
		// random values between 0.0 - 1.0
		float xValue = (float) rand() / (float) RAND_MAX;
		float yValue = (float) rand() / (float) RAND_MAX;

		// assign to vectorField
		vectorField[i][0] = xValue;
		vectorField[i][1] = yValue;
	}

#ifdef _DEBUG_
	printf("Vector[0][x]: %f\n", vectorField[0][0]);
	printf("Vector[0][y]: %f\n", vectorField[0][1]);
	printf("Vector[1][x]: %f\n", vectorField[1][0]);
	printf("Vector[1][y]: %f\n", vectorField[1][1]);
#endif

	// copy input values to device
	cudaMemcpy(dataBlock.dev_inSrc, vectorField, imageSize * 2, cudaMemcpyHostToDevice);

	// start simulation
	animBitmap.anim_and_exit((void (*)(void*, int)) anim_gpu, (void (*)(void*))anim_exit );

	// destruction
	delete[] vectorField;
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	printf("...finished!\n");
}
