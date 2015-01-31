#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <float.h>

#include <iostream>
#include <fstream>

#include "cpu_anim.cuh"
#include "utils.cuh"
#include "defines.cuh"
#include "kernel_functions.cuh"

// ****************************************************************************
// KERNEL FIELDS
// ****************************************************************************

__shared__ Vector sDots[BLOCK_WIDTH + 2][BLOCK_WIDTH + 2];

// ****************************************************************************
// KERNEL FUNCTIONS (PARALLEL EXECUTION)
// ****************************************************************************

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

// simulate inside each block
__device__ void calculateNewValue(float* newValues) {

	int tidx = threadIdx.x + 1; // thread x-coordinate inside block
	int tidy = threadIdx.y + 1; // thread y-coordinate inside block

	if (blockIdx.y == 2 && blockIdx.x == 2) {
		newValues[0] = 80.0f;
		newValues[1] = 50.0f;
	} else if (blockIdx.y == 13 && blockIdx.x == 3) {
		newValues[0] = 60.0f;
		newValues[1] = -70.0f;
	} else if (blockIdx.y == 2 && blockIdx.x == 12) {
		newValues[0] = -30.0f;
		newValues[1] = 90.0f;
	} else if (blockIdx.y == 11 && blockIdx.x == 10) {
		newValues[0] = -70.0f;
		newValues[1] = -70.0f;
	} else {

		float weight = 1.0f;

		newValues[0] = weight * sDots[tidx][tidy][0];
		newValues[1] = weight * sDots[tidx][tidy][1];

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				if (x != 0 && y != 0) {
					float neighborX = sDots[tidx + x][tidy + y][0];
					float neighborY = sDots[tidx + x][tidy + y][1];

					float angle = getAngleBetween(-x, -y, neighborX, neighborY);

					// add x-y-values of neighbor pixels
					float currentWeight = (1.5 - abs(angle) / 180.0f);
					newValues[0] += currentWeight * neighborX;
					newValues[1] += currentWeight * neighborY;
					weight += currentWeight;

					// rotate based on direction of neighbor pixels
					float relativeAngle = 0.0f;
					if (getVectorLength(neighborX, neighborY) >= 0.0f + FLT_EPSILON) {
						relativeAngle = getAngleBetween(newValues[0], newValues[1], neighborX, neighborY);
					}
					float rotationSpeed = 0.1f;
					float rotationAngle = relativeAngle * rotationSpeed / 180.0f * M_PI;
					float cosinus = cos(rotationAngle);
					float sinus = sin(rotationAngle);
					newValues[0] = (newValues[0] * cosinus - newValues[1] * sinus);
					newValues[1] = (newValues[0] * sinus + newValues[1] * cosinus);
				}
			}
		}

		newValues[0] /= weight;
		newValues[1] /= weight;
	}
}

// copy input data to shared memory
__device__ void copyToSharedMem(float* inVelocityField) {

	int x = threadIdx.x;
	int y = threadIdx.y;

	float initalBorderValue = 0.0f;

	int xPos = blockIdx.x * blockDim.x * 2 + x;

	sDots[x / 2 + 1][y + 1][x % 2] = inVelocityField[(blockIdx.y * 32 + y) * GESAMTBREITE + xPos];
	sDots[x / 2 + 1 + 16][y + 1][x % 2] = inVelocityField[(blockIdx.y * 32 + y) * GESAMTBREITE + xPos + 32];

	// not in first row
	if (blockIdx.y != 0) {
		if (y == 0)
			sDots[x / 2 + 1][0][x % 2] = inVelocityField[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos];
		else if (y == 1)
			sDots[x / 2 + 1 + 16][0][x % 2] = inVelocityField[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos + 32];
		else if (y == 2 && (x == 0 || x == 1))
			sDots[0][0][(x + 1) % 2] =
					blockIdx.x != 0 ?
							inVelocityField[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos - (x * 2 + 1)] :
							initalBorderValue;
		else if (y == 3 && (x == 0 || x == 1))
			sDots[33][0][x] =
					blockIdx.x != (512 / 32 - 1) ?
							inVelocityField[(blockIdx.y * 32 - 1) * GESAMTBREITE + xPos + (32 * 2)] : initalBorderValue;
	} else {
		if (y == 0)
			sDots[x / 2 + 1][0][x % 2] = initalBorderValue;
		else if (y == 1)
			sDots[x / 2 + 1 + 16][0][x % 2] = initalBorderValue;
		else if (y == 2 && (x == 0 || x == 1))
			sDots[0][0][x] = initalBorderValue;
		else if (y == 3 && (x == 0 || x == 1))
			sDots[33][0][x] = initalBorderValue;
	}

	// not in last row (?) access violation (?)
	if (blockIdx.y != (512 / 32 - 1)) {
		if (y == 4)
			sDots[x / 2 + 1][33][x % 2] = inVelocityField[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos];
		else if (y == 5)
			sDots[x / 2 + 1 + 16][33][x % 2] = inVelocityField[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos + 32];
		else if (y == 6 && (x == 0 || x == 1))
			sDots[0][33][(x + 1) % 2] =
					blockIdx.x != 0 ?
							inVelocityField[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos - (x * 2 + 1)] :
							initalBorderValue;
		else if (y == 7 && (x == 0 || x == 1))
			sDots[33][33][x] =
					blockIdx.x != (512 / 32 - 1) ?
							inVelocityField[((blockIdx.y + 1) * 32) * GESAMTBREITE + xPos + (32 * 2)] :
							initalBorderValue;
	} else {
		if (y == 4)
			sDots[x + 1][33][0] = 0;
		else if (y == 5)
			sDots[x + 1][33][1] = 0;
		else if (y == 6 && (x == 0 || x == 1))
			sDots[0][33][x] = 0;
		else if (y == 7 && (x == 0 || x == 1))
			sDots[33][33][x] = 0;
	}

	// not in first column
	if (blockIdx.x != 0) {
		if (y == 8)
			sDots[0][x / 2 + 1][(x + 1) % 2] = inVelocityField[(blockIdx.y * 32) * GESAMTBREITE + (x / 2) * GESAMTBREITE
					+ blockIdx.x * 64 - (1 + x % 2)];
		if (y == 9)
			sDots[0][x / 2 + 1 + 16][(x + 1) % 2] = inVelocityField[(blockIdx.y * 32) * GESAMTBREITE
					+ (x / 2 + 16) * GESAMTBREITE + blockIdx.x * 64 - (1 + x % 2)];

	} else {
		if (y == 8)
			sDots[0][x + 1][0] = 0;
		if (y == 9)
			sDots[0][x + 1][1] = 0;
	}

	// not in last column
	if (blockIdx.x != (512 / 32 - 1)) {
		if (y == 10)
			sDots[33][x / 2 + 1][x % 2] = inVelocityField[(blockIdx.y * 32) * GESAMTBREITE + (x / 2) * GESAMTBREITE
					+ blockIdx.x * 64 + x % 2 + 64];
		if (y == 11)
			sDots[33][x / 2 + 1 + 16][x % 2] = inVelocityField[(blockIdx.y * 32) * GESAMTBREITE
					+ (x / 2 + 16) * GESAMTBREITE + blockIdx.x * 64 + x % 2 + 64];
	} else {
		if (y == 10)
			sDots[33][x + 1][0] = 0;
		if (y == 11)
			sDots[33][x + 1][1] = 0;
	}

	__syncthreads();
}

// simulation function (will be called once per run loop)
__global__ void simulate(float* g_inVelocityField, float* g_outVelocityField) {

	int offset = getArrayOffset();
	copyToSharedMem(g_inVelocityField);
	float newValues[2];
	calculateNewValue(newValues);
	g_outVelocityField[offset] = newValues[0];
	g_outVelocityField[offset + 1] = newValues[1];
}

// ****************************************************************************
// HOST FUNCTIONS (SERIAL EXECUTION)
// ****************************************************************************

float getPixelSerial(int x, int y, int pos, float* values) {

	if (x < 0 || y < 0 || x >= SIM_WIDTH || y >= SIM_HEIGHT)
		return 0;

	return values[(y * SIM_WIDTH + x) * 2 + pos];
}

void calculateNewValuesSerial(float* inVelocityField, float* outVelocityField) {

	Vector newValues;
	float weight;

	for (int yM = 0; yM < SIM_HEIGHT; yM++) {
		for (int xM = 0; xM < SIM_WIDTH; xM++) {

			if (yM / 32 == 2 && xM / 32 == 2) {
				newValues[0] = 80.0f;
				newValues[1] = 50.0f;
			} else if (yM / 32 == 13 && xM / 32 == 3) {
				newValues[0] = 60.0f;
				newValues[1] = -70.0f;
			} else if (yM / 32 == 2 && xM / 32 == 12) {
				newValues[0] = -30.0f;
				newValues[1] = 90.0f;
			} else if (yM / 32 == 11 && xM / 32 == 10) {
				newValues[0] = -70.0f;
				newValues[1] = -70.0f;
			} else {

				weight = 1.0f;

				newValues[0] = weight * getPixelSerial(xM, yM, 0, inVelocityField);
				newValues[1] = weight * getPixelSerial(xM, yM, 1, inVelocityField);

				for (int x = -1; x <= 1; x++) {
					for (int y = -1; y <= 1; y++) {
						if (x != 0 && y != 0) {

							float neighborX = getPixelSerial(xM + x, yM + y, 0, inVelocityField);
							float neighborY = getPixelSerial(xM + x, yM + y, 1, inVelocityField);

							float angle = getAngleBetweenSerial(-x, -y, neighborX, neighborY);

							// add x-y-inVelocityField of neighbor pixels
							float currentWeight = (1.5 - abs(angle) / 180.0f);
							newValues[0] += currentWeight * neighborX;
							newValues[1] += currentWeight * neighborY;
							weight += currentWeight;

							// rotate based on direction of neighbor pixels
							float relativeAngle = 0.0f;
							if (getVectorLengthSerial(neighborX, neighborY) >= 0.0f + FLT_EPSILON) {
								relativeAngle = getAngleBetweenSerial(newValues[0], newValues[1], neighborX, neighborY);
							}
							float rotationSpeed = 0.1f;
							float rotationAngle = relativeAngle * rotationSpeed / 180.0f * M_PI;
							float cosinus = cos(rotationAngle);
							float sinus = sin(rotationAngle);
							newValues[0] = (newValues[0] * cosinus - newValues[1] * sinus);
							newValues[1] = (newValues[0] * sinus + newValues[1] * cosinus);
						}
					}
				}

				newValues[0] /= weight;
				newValues[1] /= weight;
			}
			outVelocityField[(yM * SIM_WIDTH + xM) * 2] = newValues[0];
			outVelocityField[(yM * SIM_WIDTH + xM) * 2 + 1] = newValues[1];
		}
	}
}

void simulateSerial(float* inVelocityField, float* outVelocityField) {
	calculateNewValuesSerial(inVelocityField, outVelocityField);
}

// ****************************************************************************
// ANIMATION FUNCTIONS
// ****************************************************************************

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

#define __CUDA__

#ifdef __CUDA__

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	simulate<<<blocks, threads>>>(d->dev_inSrc, d->dev_outSrc);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Simulate: %f ms\n", time);

	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
	swap(d->dev_inSrc, d->dev_outSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
#else // serial execution
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	simulateSerial(d->dev_inSrc, d->dev_outSrc);

	float_to_color_serial(d->output_bitmap, d->dev_inSrc);
	swap(d->dev_inSrc, d->dev_outSrc);
	memcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Simulate: %f ms\n", time);
#endif

}

// ****************************************************************************
// MAIN
// ****************************************************************************

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

#ifdef __CUDA__
	cudaMalloc((void**) &dataBlock.output_bitmap, imageSize * 2);
	cudaMalloc((void**) &dataBlock.dev_inSrc, imageSize * 2);
	cudaMalloc((void**) &dataBlock.dev_outSrc, imageSize * 2);
#else
	dataBlock.output_bitmap = (unsigned char*) malloc(imageSize * 2);
	dataBlock.dev_inSrc = (float*) malloc(imageSize * 2);
	dataBlock.dev_outSrc = (float*) malloc(imageSize * 2);
#endif

	// initialize vectorField
	for (int i = 0; i < SIM_HEIGHT * SIM_WIDTH; i++) {
		// circle values
//		float xValue = cos(((float) i / (SIM_WIDTH * SIM_HEIGHT - 1)) * 2 * M_PI);
//		float yValue = sin(((float) i / (SIM_WIDTH * SIM_HEIGHT - 1)) * 2 * M_PI);

		// assign to vectorField
		velocityField[i][0] = 0;
		velocityField[i][1] = 0;
	}

#ifdef __CUDA__
	// copy input values to device
	cudaMemcpy(dataBlock.dev_inSrc, velocityField, imageSize * 2, cudaMemcpyHostToDevice);
#else
	memcpy(dataBlock.dev_inSrc, velocityField, imageSize * 2);
#endif

	// start simulation
	animBitmap.anim_and_exit((void (*)(void*, int)) anim_gpu, (void (*)(void*))anim_exit );
}
