/*
 * defines.cuh
 *
 *  Created on: 18.01.2015
 *      Author: thomasirmer
 */

#ifndef DEFINES_CUH_
#define DEFINES_CUH_

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


#endif /* DEFINES_CUH_ */
