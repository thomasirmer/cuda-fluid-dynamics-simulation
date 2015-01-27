//#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "cpu_anim.cuh"
#include "utils.cuh"
#include "defines.cuh"
#include "kernel_functions.cuh"

#define BLOCK_WIDTH 32

#define BLOCKBREITE  BLOCK_WIDTH
#define GESAMTBREITE 1024

//#define _DEBUG_

// KERNEL FIELDS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
__shared__ Vector sVelocityField[BLOCK_WIDTH + 2][BLOCK_WIDTH + 2];

// KERNEL FUNCTIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// global threadID for 2D grid of 2D blocks
__device__ int getGlobalThreadId() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	return threadId;
}

// offset for 1-dimensional array index
__device__ int getArrayOffset() {
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int yOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = (xOffset + yOffset * blockDim.x * gridDim.x) * 2;

	return offset;
}

// offset for 1-dimensional array index for block above
__device__ int getArrayOffsetAbove() {
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int yOffset = blockDim.y - 1 + blockIdx.y - 1 * blockDim.y;
	int offset = (xOffset + yOffset * blockDim.x * gridDim.x) * 2;

	return offset;
}

// offset for 1-dimensional array index for block below
__device__ int getArrayOffsetBelow() {
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int yOffset = 0 + blockIdx.y + 1 * blockDim.y;
	int offset = (xOffset + yOffset * blockDim.x * gridDim.x) * 2;

	return offset;
}

// simulate inside each block
__device__ void calculateNewValue(float* res) {

	int tidx = threadIdx.x + 1; // thread x-coordinate inside block
	int tidy = threadIdx.y + 1; // thread y-coordinate inside block

	if (blockIdx.y == 2 && blockIdx.x == 7) {
		res[0] = 250000.0f;
		res[1] = 250500.0f;
	} else {
		res[0] = sVelocityField[tidx][tidy][0];
		res[1] = sVelocityField[tidx][tidy][1];

		float weight = 1.0f;

		for (int x = -1; x <= 1; x++)
			for (int y = -1; y <= 1; y++)
				if (x != 0 && y != 0) {
					float angle = getAngleBetween(x, y, sVelocityField[tidx + x][tidy + y][0], sVelocityField[tidx + x][tidy + y][1]);
					float currentWeight = (1.5f - angle / 90.0f);
					res[0] += currentWeight * sVelocityField[tidx + x][tidy + y][0];
					res[1] += currentWeight * sVelocityField[tidx + x][tidy + y][1];
					weight += currentWeight;
				}

		res[0] /= weight;
		res[1] /= weight;
	}
}

// copy input data to shared memory
__device__ void copyToSharedMem(float* inVelocityField) {

	int x = threadIdx.x;
	int y = threadIdx.y;

	float initalBorderValue = 0.0f;

	int xPos = blockIdx.x * blockDim.x * 2 + x;

	sVelocityField[x / 2 + 1][y + 1][x % 2] = inVelocityField[(blockIdx.y * 32 + y) * GESAMTBREITE + xPos];
	sVelocityField[x / 2 + 1 + 16][y + 1][x % 2] = inVelocityField[(blockIdx.y * 32 + y) * GESAMTBREITE + xPos + 32];

	// not in first row
	if (blockIdx.y != 0) {
		if (y == 0)
			sVelocityField[x / 2 + 1][0][x % 2] = inVelocityField[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos];
		else if (y == 1)
			sVelocityField[x / 2 + 1 + 16][0][x % 2] = inVelocityField[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos + 32];
		else if (y == 2 && (x == 0 || x == 1))
			sVelocityField[0][0][(x + 1) % 2] = blockIdx.x != 0 ? inVelocityField[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos - (x * 2 + 1)] : initalBorderValue;
		else if (y == 3 && (x == 0 || x == 1))
			sVelocityField[33][0][x] = blockIdx.x != (512 / 32 - 1) ? inVelocityField[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos + (32 * 2)] : initalBorderValue;
	} else {
		if (y == 0)
			sVelocityField[x / 2 + 1][0][x % 2] = initalBorderValue;
		else if (y == 1)
			sVelocityField[x / 2 + 1 + 16][0][x % 2] = initalBorderValue;
		else if (y == 2 && (x == 0 || x == 1))
			sVelocityField[0][0][x] = initalBorderValue;
		else if (y == 3 && (x == 0 || x == 1))
			sVelocityField[33][0][x] = initalBorderValue;
	}

	// not in last row (?) access violation (?)
	if (blockIdx.y != (512 / 32 - 1)) {
		if (y == 4)
			sVelocityField[x / 2 + 1][33][x % 2] = inVelocityField[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos];
		else if (y == 5)
			sVelocityField[x / 2 + 1 + 16][33][x % 2] = inVelocityField[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos + 32];
		else if (y == 6 && (x == 0 || x == 1))
			sVelocityField[0][33][(x + 1) % 2] = blockIdx.x != 0 ? inVelocityField[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos - (x * 2 + 1)] : initalBorderValue;
		else if (y == 7 && (x == 0 || x == 1))
			sVelocityField[33][33][x] = blockIdx.x != (512 / 32 - 1) ? inVelocityField[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos + (32 * 2)] : initalBorderValue;
	} else {
		if (y == 4)
			sVelocityField[x + 1][33][0] = 0;
		else if (y == 5)
			sVelocityField[x + 1][33][1] = 0;
		else if (y == 6 && (x == 0 || x == 1))
			sVelocityField[0][33][x] = 0;
		else if (y == 7 && (x == 0 || x == 1))
			sVelocityField[33][33][x] = 0;
	}

	// not in first column
	if (blockIdx.x != 0) {
		if (y1 == 8)
			sVelocityField[0][x / 2 + 1][(x + 1) % 2] = inVelocityField[(blockIdx.y * 32) * GESAMTBREITE + (x / 2) * GESAMTBREITE + blockIdx.x * 64
					- (1 + x % 2)];
		if (y == 9)
			sVelocityField[0][x / 2 + 1 + 16][(x + 1) % 2] = inVelocityField[(blockIdx.y * 32) * GESAMTBREITE + (x / 2 + 16) * GESAMTBREITE + blockIdx.x * 64
					- (1 + x % 2)];

	} else {
		if (y == 8)
			sVelocityField[0][x + 1][0] = 0;
		if (y == 9)
			sVelocityField[0][x + 1][1] = 0;
	}

	// not in last column
	if (blockIdx.x != (512 / 32 - 1)) {
		if (y1 == 10)
			sVelocityField[33][x / 2 + 1][x % 2] =
					inVelocityField[(blockIdx.y * 32) * GESAMTBREITE + (x / 2) * GESAMTBREITE + blockIdx.x * 64 + x % 2 + 64];
		if (y == 11)
			sVelocityField[33][x / 2 + 1 + 16][x % 2] = inVelocityField[(blockIdx.y * 32) * GESAMTBREITE + (x / 2 + 16) * GESAMTBREITE + blockIdx.x * 64
					+ x % 2 + 64];
	} else {
		if (y == 10)
			sVelocityField[33][x + 1][0] = 0;
		if (y == 11)
			sVelocityField[33][x + 1][1] = 0;
	}

	__syncthreads();
}

// simulation function (will be called once per run loop)
__global__ void simulate(float* inVelocityField, float* outVelocityField) {

	int offset = getArrayOffset();

	copyToSharedMem(inVelocityField);

	float res[2];
	calculateNewValue(res);

	outVelocityField[offset] = res[0];
	outVelocityField[offset + 1] = res[1];
}

// ANIMATION FUNCTIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
	dim3 blocks(ceil(SIM_WIDTH / BLOCK_WIDTH), ceil(SIM_HEIGHT / BLOCK_WIDTH));
	dim3 threads(BLOCK_WIDTH, BLOCK_WIDTH);
	CPUAnimBitmap* bitmap = d->bitmap;

	//copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
	simulate<<<blocks, threads>>>(d->dev_inSrc, d->dev_outSrc);
	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_outSrc);
	swap(d->dev_inSrc, d->dev_outSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);

//	usleep(1000000);
}

// MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
int main(void) {
	printf("Starting CUDA-Application - Parallel Fluid Simulation ...\n");

	// initialize data field which will be used for all further calculation
	Vector* velocityField = new Vector[SIM_HEIGHT * SIM_WIDTH];

	// set up stuff for graphical output
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
		velocityField[i][0] = 0;
		velocityField[i][1] = 0;
	}

//	vectorField[90000][0] = 25000;
//	vectorField[90000][1] = 0;

	// copy input values to device
	cudaMemcpy(dataBlock.dev_inSrc, velocityField, imageSize * 2, cudaMemcpyHostToDevice);

	// start simulation
	animBitmap.anim_and_exit((void (*)(void*, int)) anim_gpu, (void (*)(void*))anim_exit );
}
