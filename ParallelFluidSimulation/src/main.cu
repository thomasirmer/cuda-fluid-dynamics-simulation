#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include "cpu_anim.cuh"
#include "utils.cuh"
#include "defines.cuh"

#include "kernel_functions.cuh"

#define BLOCK_WIDTH 32

#define BLOCKBREITE  BLOCK_WIDTH
#define GESAMTBREITE SIM_WIDTH

//#define _DEBUG_

// KERNEL FIELDS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
__shared__ Vector dots[BLOCK_WIDTH + 2][BLOCK_WIDTH + 2];

// KERNEL FUNCTIONS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// global threadID for 2D grid of 2D blocks
__device__ int getGlobalThreadId() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	return threadId;
}

__device__ int getArrayOffset() {
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int yOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = (xOffset + yOffset * blockDim.x * gridDim.x) * 2;

	return offset;
}

__device__ int getArrayOffsetAbove() {
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int yOffset = blockDim.y - 1 + blockIdx.y - 1 * blockDim.y;
	int offset = (xOffset + yOffset * blockDim.x * gridDim.x) * 2;

	return offset;
}

__device__ int getArrayOffsetBelow() {
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int yOffset = 0 + blockIdx.y + 1 * blockDim.y;
	int offset = (xOffset + yOffset * blockDim.x * gridDim.x) * 2;

	return offset;
}

// simulate inside each block
__device__ void calculateNewValue(float* res) {

	int x = threadIdx.x % 32 + 1; // thread x-coordinate inside block
	int y = threadIdx.y % 32 + 1; // thread y-coordinate inside block

	res[0] = dots[x][y][0];
	res[1] = dots[x][y][1];

	res[0] += dots[x - 1][y][0];
	res[1] += dots[x - 1][y][1];

	res[0] += dots[x + 1][y][0];
	res[1] += dots[x + 1][y][1];

	res[0] += dots[x][y - 1][0];
	res[1] += dots[x][y - 1][1];

	res[0] += dots[x][y + 1][0];
	res[1] += dots[x][y + 1][1];

	res[0] /= 5.0f;
	res[1] /= 5.0f;

	__syncthreads();

//	for (int yi = -1; yi < 2; yi++) {
//		for (int xi = -1; xi < 2; xi++) {
//			if (!(xi == 0 && yi == 0)) {
//				res[0] += dots[x + xi][y + yi][0];
//				res[1] += dots[x + xi][y + yi][1];
//			}
//		}
//	}
//
//	res[0] /= 9;
//	res[1] /= 9;
}

// copy input data to shared memory
__device__ void copyToSharedMem(float* inDots) {

	int offset   = getArrayOffset();

	int x = threadIdx.x % 32 + 1; // thread x-coordinate inside block
	int y = threadIdx.y % 32 + 1; // thread y-coordinate inside block

	dots[x][y][0] = inDots[offset];		// copy x-values from global to shared memory
	dots[x][y][1] = inDots[offset + 1];	// copy y-values from global to shared memory

	__syncthreads();

	bool isBorder = blockIdx.y == 0 || blockIdx.y == gridDim.y - 1 || blockIdx.x == 0 || blockIdx.x == gridDim.x - 1;
	float initalValue = 0.0f; // initial value for border-pixels

	if (blockIdx.y == 0) { // top border blocks
		if (threadIdx.y == 0) { // top border pixels
			dots[x][y - 1][0] = initalValue;
			dots[x][y - 1][1] = initalValue;
		} else if (threadIdx.y == blockDim.y - 1) { // pixels from block below
			dots[x][y + 1][0] = inDots[offset - gridDim.x * blockDim.x * 2];
			dots[x][y + 1][1] = inDots[offset + 1 - gridDim.x * blockDim.x * 2 + 1];
		}
	} else if (blockIdx.y == gridDim.y - 1) { // bottom border blocks
		if (threadIdx.y == blockDim.y - 1) { // bottom border pixels
			dots[x][y + 1][0] = initalValue;
			dots[x][y + 1][1] = initalValue;
		} else if (threadIdx.y == 0) { // pixels from block above
			dots[x][y - 1][0] = inDots[offset + gridDim.x * blockDim.x * 2];
			dots[x][y - 1][1] = inDots[offset + 1 + gridDim.x * blockDim.x * 2 + 1];
		}
	} else if (blockIdx.x == 0) { // left border blocks
		if (threadIdx.x == 0) { // left border pixels
			dots[x - 1][y][0] = initalValue;
			dots[x - 1][y][1] = initalValue;
		} else if (threadIdx.x == blockDim.x - 1) { // pixels from right-side block
			dots[x + 1][y][0] = inDots[offset + 2];
			dots[x + 1][y][1] = inDots[offset + 1 + 2];
		}
	} else if (blockIdx.x == gridDim.x - 1) { // right border blocks
		if (threadIdx.x == blockDim.x - 1) { // right border pixels
			dots[x + 1][y][0] = initalValue;
			dots[x + 1][y][1] = initalValue;
		} else if (threadIdx.x == 0) { // pixels from left-side block
			dots[x - 1][y][0] = inDots[offset - 2];
			dots[x - 1][y][1] = inDots[offset + 1 - 2];
		}
	}

	// all non-border blocks
	if (!isBorder) {
		if (threadIdx.y == 0) { // top pixels
			dots[x][y - 1][0] = inDots[offset + gridDim.x * blockDim.x * 2];
			dots[x][y - 1][1] = inDots[offset + 1 + gridDim.x * blockDim.x * 2 + 1];
		} else if (threadIdx.y == blockDim.y - 1) { // bottom pixels
			dots[x][y + 1][0] = inDots[offset - gridDim.x * blockDim.x * 2];
			dots[x][y + 1][1] = inDots[offset + 1 - gridDim.x * blockDim.x * 2 + 1];
		} else if (threadIdx.x == 0) { // left pixels
			dots[x - 1][y][0] = inDots[offset - 2];
			dots[x - 1][y][1] = inDots[offset + 1 - 2];
		} else if (threadIdx.x == blockDim.x - 1) { // right pixels
			dots[x + 1][y][0] = inDots[offset + 2];
			dots[x + 1][y][1] = inDots[offset + 1 + 2];
		}
	}

	__syncthreads();
}

// simulation function (will be called once per run loop)
__global__ void simulate(float* inDots, float* outDots) {

	int threadID = getGlobalThreadId();
	int offset	 = getArrayOffset();

	copyToSharedMem(inDots);

	float res[2];
	calculateNewValue(res);

	outDots[offset] = res[0];
	outDots[offset + 1] = res[1];
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
	Vector* vectorField = new Vector[SIM_HEIGHT * SIM_WIDTH];

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
		vectorField[i][0] = xValue;
		vectorField[i][1] = yValue;
	}

	// copy input values to device
	cudaMemcpy(dataBlock.dev_inSrc, vectorField, imageSize * 2, cudaMemcpyHostToDevice);

	// start simulation
	animBitmap.anim_and_exit((void (*)(void*, int)) anim_gpu, (void (*)(void*))anim_exit );
}
