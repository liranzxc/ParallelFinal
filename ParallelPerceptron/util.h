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

	double * values;
};

void MyCudaMalloc(void** dev_pointer, size_t size, int error_label);

void MyCudaCopy(void* dest, void * src, size_t size, cudaMemcpyKind kind, int error_label);

void FreeConstanstCuda(Point * dev_pts, double * dev_values, int * dev_n, int * dev_k);

void MyCudaFree(void * object,int error_label);
cudaError_t FreeFunction( double * dev_W,double  * dev_alfa, int * dev_mislead,int * dev_tempresult);
double ProcessAlfa(Point * dev_pts, double* dev_values ,double  * alfa, int * dev_n, 
	int  * dev_k, int limit, double QC,int n ,int k,double ** WSaved);



__device__ double dot(double * dev_w, double * dev_x, int indexValues, int * dev_k);

__global__ void createNewWeight(double * dev_alfa, double *dev_values, int * indexerValues, double * W_dev);


void DoJob(double alfaZero, double alfaMax,double step, Point * dev_pts, double * dev_values, int * dev_n, int * dev_k, int n, int k, int limit,
	double QC, double * qMin, double * wMin, double *alfaMin);
void mallocConstastCuda(Point * pts, int n, int k, Point ** dev_pts, int ** dev_n, int ** dev_k, double ** dev_values);