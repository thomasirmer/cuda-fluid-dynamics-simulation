/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#ifndef __BOOK_H__
#define __BOOK_H__
#include <stdio.h>

#include "defines.cuh"
#include "kernel_functions.cuh"
#include <math.h>

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit (EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

template<typename T>
void swap(T& a, T& b) {
	T t = a;
	a = b;
	b = t;
}

void* big_random_block(int size) {
	unsigned char *data = (unsigned char*) malloc(size);
	HANDLE_NULL(data);
	for (int i = 0; i < size; i++)
		data[i] = rand();

	return data;
}

int* big_random_block_int(int size) {
	int *data = (int*) malloc(size * sizeof(int));
	HANDLE_NULL(data);
	for (int i = 0; i < size; i++)
		data[i] = rand();

	return data;
}

// a place for common kernels - starts here

__device__ unsigned char value(float n1, float n2, int hue) {
	if (hue > 360)
		hue -= 360;
	else if (hue < 0)
		hue += 360;

	if (hue < 60)
		return (unsigned char) (255 * (n1 + (n2 - n1) * hue / 60));
	if (hue < 180)
		return (unsigned char) (255 * n2);
	if (hue < 240)
		return (unsigned char) (255 * (n1 + (n2 - n1) * (240 - hue) / 60));
	return (unsigned char) (255 * n1);
}

__global__ void float_to_color(unsigned char *outBitmap, const float* vectorValues) {
	int threadID = getGlobalThreadId();

	if (threadID < SIM_WIDTH * SIM_HEIGHT) {
		float xValue = vectorValues[threadID * 2];
		float yValue = vectorValues[threadID * 2 + 1];

		float angleRad = getVectorAngle(xValue, yValue);
		unsigned int angleDeg = (unsigned int) round(angleRad + 180.0f);
		unsigned int length = (unsigned int) floor(getVectorLength(xValue, yValue) * 255);

		unsigned char red, green, blue;
		hsv2rgb(angleDeg, 255, length, &red, &green, &blue, 255);

		// red - green - blue - (?) alpha (?)
		outBitmap[threadID * 4 + 0] = red;
		outBitmap[threadID * 4 + 1] = green;
		outBitmap[threadID * 4 + 2] = blue;
		outBitmap[threadID * 4 + 3] = 255;
	}
}

void float_to_color_serial(unsigned char *outBitmap, const float* vectorValues) {

	for (int y = 0; y < SIM_HEIGHT; y++)
		for (int x = 0; x < SIM_WIDTH; x++) {
			float xValue = vectorValues[(y * SIM_WIDTH + x) * 2];
			float yValue = vectorValues[(y * SIM_WIDTH + x) * 2 + 1];

			float angleRad = getVectorAngleSerial(xValue, yValue);
			unsigned int angleDeg = (unsigned int) round(angleRad + 180.0f);
			unsigned int length = (unsigned int) floor(getVectorLengthSerial(xValue, yValue) * 255);

			unsigned char red, green, blue;
			hsv2rgbSerial(angleDeg, 255, length, &red, &green, &blue, 255);

			// red - green - blue - (?) alpha (?)
			outBitmap[(y * SIM_WIDTH + x) * 4 + 0] = red;
			outBitmap[(y * SIM_WIDTH + x) * 4 + 1] = green;
			outBitmap[(y * SIM_WIDTH + x) * 4 + 2] = blue;
			outBitmap[(y * SIM_WIDTH + x) * 4 + 3] = 255;
		}
}

		__global__ void float_to_color(uchar4 *optr, const float *outSrc) {
			// map from threadIdx/BlockIdx to pixel position
			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;
			int offset = x + y * blockDim.x * gridDim.x;

			float l = outSrc[offset];
			float s = 1;
			int h = (180 + (int) (360.0f * outSrc[offset])) % 360;
			float m1, m2;

			if (l <= 0.5f)
				m2 = l * (1 + s);
			else
				m2 = l + s - l * s;
			m1 = 2 * l - m2;

			optr[offset].x = value(m1, m2, h + 120);
			optr[offset].y = value(m1, m2, h);
			optr[offset].z = value(m1, m2, h - 120);
			optr[offset].w = 255;
		}

#if _WIN32
//Windows threads.
#include <windows.h>

		typedef HANDLE CUTThread;
		typedef unsigned (WINAPI *CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC unsigned WINAPI
#define  CUT_THREADEND return 0

#else
//POSIX threads.
#include <pthread.h>

		typedef pthread_t CUTThread;
		typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define  CUT_THREADEND
#endif

//Create thread.
		CUTThread start_thread(CUT_THREADROUTINE, void *data);

//Wait for thread to finish.
		void end_thread(CUTThread thread);

//Destroy thread.
		void destroy_thread(CUTThread thread);

//Wait for multiple threads.
		void wait_for_threads(const CUTThread *threads, int num);

#if _WIN32
//Create thread
		CUTThread start_thread(CUT_THREADROUTINE func, void *data) {
			return CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, data, 0, NULL);
		}

//Wait for thread to finish
		void end_thread(CUTThread thread) {
			WaitForSingleObject(thread, INFINITE);
			CloseHandle(thread);
		}

//Destroy thread
		void destroy_thread( CUTThread thread ) {
			TerminateThread(thread, 0);
			CloseHandle(thread);
		}

//Wait for multiple threads
		void wait_for_threads(const CUTThread * threads, int num) {
			WaitForMultipleObjects(num, threads, true, INFINITE);

			for(int i = 0; i < num; i++)
			CloseHandle(threads[i]);
		}

#else
//Create thread
		CUTThread start_thread(CUT_THREADROUTINE func, void * data) {
			pthread_t thread;
			pthread_create(&thread, NULL, func, data);
			return thread;
		}

//Wait for thread to finish
		void end_thread(CUTThread thread) {
			pthread_join(thread, NULL);
		}

//Destroy thread
		void destroy_thread(CUTThread thread) {
			pthread_cancel(thread);
		}

//Wait for multiple threads
		void wait_for_threads(const CUTThread * threads, int num) {
			for (int i = 0; i < num; i++)
				end_thread(threads[i]);
		}

#endif

#endif  // __BOOK_H__
