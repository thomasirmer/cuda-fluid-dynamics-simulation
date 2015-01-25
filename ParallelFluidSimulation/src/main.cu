#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cpu_anim.cuh"
#include "utils.cuh"
#include "defines.cuh"

#include "kernel_functions.cuh"

#define SIM_WIDTH   512
#define SIM_HEIGHT  512
#define NUM_THREADS 32

#define BLOCKBREITE  NUM_THREADS
#define GESAMTBREITE SIM_WIDTH

//#define _DEBUG_

// KERNEL FUNCTIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
__shared__ Vector dots[BLOCKBREITE + 2][BLOCKBREITE + 2];

__device__ void calculateNewValue(float* res) {
	int threadID = getGlobalThreadId();

	int x = threadIdx.x % 32 + 1;
	int y = threadIdx.y % 32 + 1;

	res[0] = dots[x][y][0];
	res[1] = dots[x][y][1];

	for (int yi = -1; yi < 2; yi++) {
		for (int xi = -1; xi < 2; xi++) {
			if (!(xi == 0 && yi == 0)) {
				res[0] += dots[x + xi][y + yi][0];
				res[1] += dots[x + xi][y + yi][1];
			}
		}
	}

	res[0] /= 9;
	res[1] /= 9;
}

__device__ void copyToSharedMem(float* inDots) {

	int threadID = getGlobalThreadId();

	int x1 = threadIdx.x % 32;
	int y1 = threadIdx.y % 32;

	float initalValue = 0.0f;

	dots[x1][y1][0] = inDots[threadID * 2];
	dots[x1][y1][1] = inDots[threadID * 2 + 1];

	int xPos = blockIdx.x * blockDim.x * 2 + x1;

	// Wir befinden uns nicht in der ersten Reihe
	if (blockIdx.y != 0) {
		if (y1 == 0)
			dots[x1 + 1][0][x1 % 2] = inDots[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos];
		else if (y1 == 1)
			dots[x1 + 1 + 16][0][x1 % 2] = inDots[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos + 32];
		else if (y1 == 2 && (x1 == 0 || x1 == 1))
			dots[0][0][(x1 + 1) % 2] = inDots[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos - (x1 * 2 + 1)];
		else if (y1 == 3 && (x1 == 0 || x1 == 1))
			dots[33][0][x1] = inDots[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos + (32 * 2)];
	} else {
		if (y1 == 0)
			dots[x1 + 1][0][x1 % 2] = initalValue;
		else if (y1 == 1)
			dots[x1 + 1 + 16][0][x1 % 2] = initalValue;
		else if (y1 == 2 && (x1 == 0 || x1 == 1))
			dots[0][0][x1] = initalValue;
		else if (y1 == 3 && (x1 == 0 || x1 == 1))
			dots[33][0][x1] = initalValue;
	}

	// Nicht in der letzten Reihe
	if (blockIdx.y != 512 / 32 - 1) {
		if (y1 == 4)
			dots[x1 + 1][33][x1 % 2] = inDots[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos];
		else if (y1 == 5)
			dots[x1 + 1 + 16][33][x1 % 2] = inDots[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos + 32];
		else if (y1 == 6 && (x1 == 0 || x1 == 1))
			dots[0][33][(x1 + 1) % 2] = inDots[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos - (x1 * 2 + 1)];
		else if (y1 == 7 && (x1 == 0 || x1 == 1))
			dots[33][33][x1] = inDots[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos + (32 * 2)];
	} else {
		if (y1 == 4)
			dots[x1 + 1][33][0] = 0;
		else if (y1 == 5)
			dots[x1 + 1][33][1] = 0;
		else if (y1 == 6 && (x1 == 0 || x1 == 1))
			dots[0][33][x1] = 0;
		else if (y1 == 7 && (x1 == 0 || x1 == 1))
			dots[33][33][x1] = 0;
	}

	// Nicht erste Spalte
	if (blockIdx.x != 0) {
		if (y1 == 8)
			dots[0][x1 + 1][(x1 + 1) % 2] = inDots[(blockIdx.y * 32) * GESAMTBREITE + (x1 / 2) * GESAMTBREITE + blockIdx.x * 64 - (1 + x1 % 2)];
		if (y1 == 9)
			dots[0][x1 + 1 + 16][(x1 + 1) % 2] = inDots[(blockIdx.y * 32) * GESAMTBREITE + (x1 / 2 + 16) * GESAMTBREITE + blockIdx.x * 64 - (1 + x1 % 2)];

	} else {
		if (y1 == 8)
			dots[0][x1 + 1][0] = 0;
		if (y1 == 9)
			dots[0][x1 + 1][1] = 0;
	}

	if (blockIdx.x == 512 / 32 - 1) {
		if (y1 == 10)
			dots[33][x1 + 1][x1 % 2] = inDots[(blockIdx.y * 32) * GESAMTBREITE + (x1 / 2) * GESAMTBREITE + blockIdx.x * 64 + x1 % 2 + 64];
		if (y1 == 11)
			dots[33][x1 + 1 + 16][x1 % 2] = inDots[(blockIdx.y * 32) * GESAMTBREITE + (x1 / 2 + 16) * GESAMTBREITE + blockIdx.x * 64 + x1 % 2 + 64];
	} else {
		if (y1 == 10)
			dots[33][x1 + 1][0] = 0;
		if (y1 == 11)
			dots[33][x1 + 1][1] = 0;
	}

	__syncthreads();

}

__global__ void simulate(float* inDots, float* outDots) {

	int threadID = getGlobalThreadId();

	copyToSharedMem(inDots);

	float res[2];
	calculateNewValue(res);

	outDots[threadID * 2] = res[0];
	outDots[threadID * 2 + 1] = res[1];
}

// global threadID for 2D grid of 2D blocks
__device__ int getGlobalThreadId() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	return threadId;
}

// exit function for run-loop
void anim_exit(DataBlock *d) {
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);
	cudaFree(d->bitmap);
	cudaFree(d->output_bitmap);
}

// animation run-loop function
void anim_gpu(DataBlock *d, int ticks) {
	dim3 blocks(ceil(SIM_WIDTH / NUM_THREADS), ceil(SIM_HEIGHT / NUM_THREADS));
	dim3 threads(NUM_THREADS, NUM_THREADS);
	CPUAnimBitmap* bitmap = d->bitmap;

	//copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
	simulate<<<blocks, threads>>>(d->dev_inSrc, d->dev_outSrc);
	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_outSrc);
	swap(d->dev_inSrc, d->dev_outSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
}

int main(void) {
	printf("Starting CUDA-Application - Parallel Fluid Simulation ...\n");

	// initialize data field which will be used for all further calculation
	Vector* vectorField = new Vector[SIM_HEIGHT * SIM_WIDTH];

	// set up stuff for graphical outout
	DataBlock dataBlock;
	CPUAnimBitmap animBitmap(SIM_WIDTH, SIM_HEIGHT, &dataBlock);
	dataBlock.bitmap = &animBitmap;

	// allocate device memory
	int imageSize = animBitmap.image_size(); // image_size() returns width * height * 4
	cudaMalloc((void**) &dataBlock.output_bitmap, imageSize * 2);
	cudaMalloc((void**) &dataBlock.dev_inSrc, imageSize * 2);
	cudaMalloc((void**) &dataBlock.dev_outSrc, imageSize * 2);

	// initialize vectorField
	for (int i = 0; i < SIM_HEIGHT * SIM_WIDTH; i++) {
		// circle values
		float xValue = cos(((float) i / (SIM_WIDTH * SIM_HEIGHT - 1)) * 2 * M_PI);
		float yValue = sin(((float) i / (SIM_WIDTH * SIM_HEIGHT - 1)) * 2 * M_PI);

		// assign to vectorField
		vectorField[i][0] = xValue;
		vectorField[i][1] = yValue;
	}

	// copy input values to device
	cudaMemcpy(dataBlock.dev_inSrc, vectorField, imageSize * 2, cudaMemcpyHostToDevice);

	// start simulation
	animBitmap.anim_and_exit((void (*)(void*, int)) anim_gpu, (void (*)(void*))anim_exit );
}
