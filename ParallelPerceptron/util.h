#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

#include <stdio.h>
#include "util.h"

#define MASTER 0
#define NUM_PROCESSES 4
struct Point
{

	int group;

	float * values;
};

void MyCudaMalloc(void** dev_pointer, size_t size, int error_label);

void MyCudaCopy(void* dest, void * src, size_t size, cudaMemcpyKind kind, int error_label);

void MyCudaFree(void * object);
cudaError_t FreeFunction( float * dev_W,float  * dev_alfa, int * dev_mislead,int * dev_tempresult);
double ProcessAlfa(Point * dev_pts, float* dev_values ,float  * alfa, int * dev_n, 
	int  * dev_k, int limit, float QC,int n ,int k,float ** WSaved);



__device__ float dot(float * dev_w, float * dev_x, int indexValues, int * dev_k);

__global__ void createNewWeight(float * dev_alfa, float *dev_values, int * indexerValues, float * W_dev);



void DoJob(float alfaZero, float alfaMax, Point * pts, int n, int k, int limit, float QC, int myrank);


