
#include <stdio.h>
#include <stdlib.h>

#include "cpu_anim.h"
#include "defines.cuh"

#define SIM_WIDTH  32
#define SIM_HEIGHT 32

void anim_exit(DataBlock *d) {
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);
}

void anim_gpu(DataBlock *d, int ticks) {
//	dim3 blocks(DIM / 16, DIM / 16);
//	dim3 threads(16, 16);
//	CPUAnimBitmap *bitmap = d->bitmap;
//
//	// i calculations before updating the bitmap
//	for (int i = 0; i < 25; i++) {
//		copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
//		blend_kernel<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
//		swap(d->dev_inSrc, d->dev_outSrc);
//	}
//
//	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
//
//	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(),
//			cudaMemcpyDeviceToHost);

	// TODO: Animation

}

int main (void) {
	printf("Starting CUDA-Application - Parallel Fluid Simulation ...\n");

	// TODO: initialize all the stuff
	//Vector2D* vectorField = new Vector2D[SIM_HEIGHT * SIM_WIDTH];
	Vector* vectorField = new Vector[SIM_HEIGHT * SIM_WIDTH];

	// TODO: do simulation (ihno)

	// TODO: visualize (thomas)
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	// random initialize vectorField
	for (int i = 0; i < SIM_HEIGHT * SIM_WIDTH; i++) {
		// random values between 0.0 - 1.0
		float xValue = (float) rand() / (float) RAND_MAX;
		float yValue = (float) rand() / (float) RAND_MAX;

		// assign to vectorField
		vectorField[i][0] = xValue;
		vectorField[i][1] = yValue;
	}

	DataBlock dataBlock;
	CPUAnimBitmap animBitmap(SIM_WIDTH, SIM_HEIGHT, &dataBlock);
	dataBlock.bitmap = &animBitmap;

	int imageSize = animBitmap.image_size();
	cudaMalloc((void**) &dataBlock.dev_inSrc,    imageSize);
	cudaMalloc((void**) &dataBlock.dev_outSrc,   imageSize);
	cudaMalloc((void**) &dataBlock.dev_constSrc, imageSize);

	// initialize constant data (border pixels are all zero)
	// top border
	for (int i = 0; i < SIM_WIDTH; i++) {
		vectorField[i][0] = 0;
		vectorField[i][1] = 0;
	}
	// bottom border
	for (int i = SIM_WIDTH * SIM_HEIGHT - SIM_WIDTH; i < SIM_WIDTH * SIM_HEIGHT; i++) {
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

	cudaMemcpy(dataBlock.dev_constSrc, vectorField, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dataBlock.dev_inSrc   , vectorField, imageSize, cudaMemcpyHostToDevice);

	animBitmap.anim_and_exit((void (*)(void*,int)) anim_gpu,
                          	 (void (*)(void*))     anim_exit );

	// destruction
	delete[] vectorField;
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	printf("...finished!\n");
}
