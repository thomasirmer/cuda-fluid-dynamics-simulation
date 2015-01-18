/*
 * defines.cuh
 *
 *  Created on: 18.01.2015
 *      Author: thomasirmer
 */

#ifndef DEFINES_CUH_
#define DEFINES_CUH_

// native data structure
typedef float Vector[2];

// data structure for simulation
struct Vector2D {
    float x;
    float y;

    // default constructor
    Vector2D() {
        	this->x = 0.0f;
        	this->y = 0.0f;
        }

    // constructor with initial values
    Vector2D(float x, float y) {
    	this->x = x;
    	this->y = y;
    }

    // destructor
    ~Vector2D() {

    }
};

// data structure needed for visualization
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
    CPUAnimBitmap   *bitmap;
};

#endif /* DEFINES_CUH_ */
