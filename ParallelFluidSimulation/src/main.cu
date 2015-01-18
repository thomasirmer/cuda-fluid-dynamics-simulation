
#include <stdio.h>
#include <stdlib.h>

#include "defines.cuh"
#include "cpu_anim.h"

#define SIM_WIDTH  32
#define SIM_HEIGHT 32

int main (void) {
	printf("Starting CUDA-Application - Parallel Fluid Simulation ...\n");

	// TODO: initialize all the stuff
	Vector2D* vectorField = new Vector2D[SIM_HEIGHT * SIM_WIDTH];

	// TODO: do simulation (ihno)

	// TODO: visualize (thomas)
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	// random initialize vectorField
	for (int i = 0; i < SIM_HEIGHT * SIM_WIDTH; i++) {
		// random values between 0.0 - 1.0
		float xValue = (float) rand() / (float) RAND_MAX;
		float yValue = (float) rand() / (float) RAND_MAX;

		// assign to vectorField
		vectorField[i].x = xValue;
		vectorField[i].y = yValue;
	}

	CPUAnimBitmap* animBitmap = new CPUAnimBitmap(SIM_WIDTH, SIM_HEIGHT, NULL);
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
}
