
#include <stdio.h>
#include <stdlib.h>

#include "defines.cuh"

#define SIM_WIDTH  512
#define SIM_HEIGHT 512

int main (void) {
	printf("Starting CUDA-Application - Parallel Fluid Simulation ...\n");

	// TODO: initialize all the stuff
	Vector2D* vectorField = (Vector2D*) malloc(sizeof(Vector2D) * SIM_HEIGHT * SIM_WIDTH);

	// TODO: do simulation (ihno)

	// TODO: visualize (thomas)
}
