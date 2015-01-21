#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cpu_anim.h"
#include "utils.h"
#include "defines.cuh"

#define SIM_WIDTH   512
#define SIM_HEIGHT  512
#define NUM_THREADS 32

#define BLOCKBREITE  NUM_THREADS
#define GESAMTBREITE SIM_WIDTH

//#define _DEBUG_

// KERNEL FUNCTIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
__shared__ Vector dots[BLOCKBREITE + 2][BLOCKBREITE + 2];

__device__ void calculateNewValue(int x1, int y1, Vector res) {
	res[0] = dots[x1][y1][0];
	res[1] = dots[x1][y1][1];

	for (int yi = -1; yi < 2; yi++) {
		for (int xi = -1; xi < 2; xi++) {
			if (!(xi == 0 && yi == 0)) {
				res[0] += dots[x1 + xi + 1][y1 + yi + 1][0];
				res[1] += dots[x1 + xi + 1][y1 + yi + 1][1];
			}
		}
	}

	res[0] /= 9;
	res[1] /= 9;
}

__device__ void copyToSharedMem(int x1, int y1, int x, int y, Vector* inDots) {

	//dots[]((float*) inDots)+(y*GESAMTBREITE)x*sizeof(float)

	dots[x1 + 1][y1 + 1][0] = inDots[y * GESAMTBREITE + x][0];
	dots[x1 + 1][y1 + 1][1] = inDots[y * GESAMTBREITE + x][1];

	// Wir befinden uns nicht in der ersten Reihe
	if (blockIdx.y != 0) {
		if (y1 == 0)
			dots[x1 + 1][0][0] =
					inDots[(blockIdx.y * 32 - 1) * GESAMTBREITE + x][0];
		else if (y1 == 1)
			dots[x1 + 1][0][1] =
					inDots[(blockIdx.y * 32 - 1) * GESAMTBREITE + x][1];
		else if (y1 == 2 && (x1 == 0 || x1 == 1))
			dots[0][0][x1] = inDots[(blockIdx.y * 32 - 1) * GESAMTBREITE + x
					- (x1 + 1)][x1];
		else if (y1 == 3 && (x1 == 0 || x1 == 1))
			dots[33][0][x1] = inDots[(blockIdx.y * 32 - 1) * GESAMTBREITE + x
					+ 31 + (1 - x1)][x1];
	} else {
		if (y1 == 0)
			dots[x1 + 1][0][0] = 0;
		else if (y1 == 1)
			dots[x1 + 1][0][1] = 0;
		else if (y1 == 2 && (x1 == 0 || x1 == 1))
			dots[0][0][x1] = 0;
		else if (y1 == 3 && (x1 == 0 || x1 == 1))
			dots[33][0][x1] = 0;
	}

	// Nicht in der letzten Reihe
	if (blockIdx.y != 512 / 32 - 1) {
		if (y1 == 4)
			dots[x1 + 1][33][0] = inDots[(blockIdx.y * 32 + 1) * GESAMTBREITE
					+ x][0];
		else if (y1 == 5)
			dots[x1 + 1][33][1] = inDots[(blockIdx.y * 32 + 1) * GESAMTBREITE
					+ x][1];
		else if (y1 == 6 && (x1 == 0 || x1 == 1))
			dots[0][33][x1] = inDots[(blockIdx.y * 32 + 1) * GESAMTBREITE + x
					- (x1 + 1)][x1];
		else if (y1 == 7 && (x1 == 0 || x1 == 1))
			dots[33][33][x1] = inDots[(blockIdx.y * 32 + 1) * GESAMTBREITE + x
					+ 31 + (1 - x1)][x1];
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
			dots[0][x1 + 1][0] = inDots[(blockIdx.y * 32) * GESAMTBREITE
					+ x1 * GESAMTBREITE + blockIdx.x * 32 - 1][0];
		if (y1 == 9)
			dots[0][x1 + 1][1] = inDots[(blockIdx.y * 32) * GESAMTBREITE
					+ x1 * GESAMTBREITE + blockIdx.x * 32 - 1][1];

	} else {
		if (y1 == 8)
			dots[0][x1 + 1][0] = 0;
		if (y1 == 9)
			dots[0][x1 + 1][1] = 0;
	}

	if (blockIdx.x == 512 / 32 - 1) {
		if (y1 == 10)
			dots[33][x1 + 1][0] = inDots[(blockIdx.y * 32) * GESAMTBREITE
					+ x1 * GESAMTBREITE + blockIdx.x * 32 + 32][0];
		if (y1 == 11)
			dots[33][x1 + 1][1] = inDots[(blockIdx.y * 32) * GESAMTBREITE
					+ x1 * GESAMTBREITE + blockIdx.x * 32 + 32][1];
	} else {
		if (y1 == 10)
			dots[33][x1 + 1][0] = 0;
		if (y1 == 11)
			dots[33][x1 + 1][1] = 0;
	}

	__syncthreads();
}

__global__ void simulate(Vector* inDots, Vector* outDots) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int x1 = x % BLOCKBREITE;
	int y1 = y % BLOCKBREITE;

	copyToSharedMem(x1, y1, x, y, inDots);

	Vector res;
	calculateNewValue(x1, y1, res);
	outDots[y * GESAMTBREITE + x][0] = res[0];
	outDots[y * GESAMTBREITE + x][0] = res[1];
}

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
//__global__ void simulate(float *outputValues, const float *inputValues) {
//	int threadId = getGlobalThreadId();
//
//	if (threadId < SIM_WIDTH * SIM_HEIGHT) {
//		outputValues[threadId * 2] = inputValues[threadId * 2];
//		outputValues[threadId * 2 + 1] = inputValues[threadId * 2 + 1];
//	}
//}

__device__ void hsv2rgb(unsigned int hue, unsigned int sat, unsigned int val,
		unsigned char * r, unsigned char * g, unsigned char * b,
		unsigned char maxBrightness) {

	unsigned int H_accent = hue / 60;
	unsigned int bottom = ((255 - sat) * val) >> 8;
	unsigned int top = val;
	unsigned char rising = ((top - bottom) * (hue % 60)) / 60 + bottom;
	unsigned char falling = ((top - bottom) * (60 - hue % 60)) / 60 + bottom;

	switch (H_accent) {
	case 0:
		*r = top;
		*g = rising;
		*b = bottom;
		break;

	case 1:
		*r = falling;
		*g = top;
		*b = bottom;
		break;

	case 2:
		*r = bottom;
		*g = top;
		*b = rising;
		break;

	case 3:
		*r = bottom;
		*g = falling;
		*b = top;
		break;

	case 4:
		*r = rising;
		*g = bottom;
		*b = top;
		break;

	case 5:
		*r = top;
		*g = bottom;
		*b = falling;
		break;
	}
	// Scale values to maxBrightness
	*r = *r * maxBrightness / 255;
	*g = *g * maxBrightness / 255;
	*b = *b * maxBrightness / 255;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

void anim_exit(DataBlock *d) {
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);
	cudaFree(d->bitmap);
	cudaFree(d->output_bitmap);
}

void anim_gpu(DataBlock *d, int ticks) {
	// TODO: Animation
	dim3 blocks(ceil(SIM_WIDTH / NUM_THREADS), ceil(SIM_HEIGHT / NUM_THREADS));
	dim3 threads(NUM_THREADS, NUM_THREADS);
	CPUAnimBitmap* bitmap = d->bitmap;

	//copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
	simulate<<<blocks, threads>>>((Vector*)d->dev_inSrc, (Vector*)d->dev_outSrc);
	swap(d->dev_inSrc, d->dev_outSrc);

	// TODO: Implement float_to_color that it uses both values!
	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_outSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(),
			cudaMemcpyDeviceToHost);
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

	// initialize vectorField
	for (int i = 0; i < SIM_HEIGHT * SIM_WIDTH; i++) {
		// random values between [0.0 ... 1.0]
		//float xValue = (float) rand() / (float) RAND_MAX;
		//float yValue = (float) rand() / (float) RAND_MAX;

		// circle values
		float xValue = cos(
				((float) i / (SIM_WIDTH * SIM_HEIGHT - 1)) * 2 * M_PI);
		float yValue = sin(
				((float) i / (SIM_WIDTH * SIM_HEIGHT - 1)) * 2 * M_PI);

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
