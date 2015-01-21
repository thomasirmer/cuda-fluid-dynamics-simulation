/*
 * defines.cuh
 *
 *  Created on: 18.01.2015
 *      Author: thomasirmer
 */

#ifndef DEFINES_CUH_
#define DEFINES_CUH_

#define SIM_WIDTH  512
#define SIM_HEIGHT 512

// native data structure
typedef float Vector[2];

// data structure needed for visualization
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
    CPUAnimBitmap   *bitmap;
};

#endif /* DEFINES_CUH_ */
