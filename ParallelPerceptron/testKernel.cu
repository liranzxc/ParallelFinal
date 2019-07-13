
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

#include <stdio.h>
#include "util.h"

#include <omp.h>

#include "testCuda.h"

void testDot();
__global__ void functionCuda(float * x, float * y, int * k);

int main()
{

	printf("hello world");
	testDot();

	return 1;
}



void testDot()
{
	cudaError_t cudaStatus;

	int k = 4;

	float * x = (float*)malloc(sizeof(float)* k);
	float * y = (float*)malloc(sizeof(float)* k);

	x[0] = 1;
	x[1] = 2;
	x[2] = 3;
	x[3] = 4;


	y[0] = 1;
	y[1] = 2;
	y[2] = 3;
	y[3] = 4;


	float * dev_x;
	float * dev_y;
	int * dev_k;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	MyCudaMalloc((void**)&dev_k, sizeof(int), 8);


	MyCudaMalloc((void**)&dev_x, sizeof(float)* k, 1);

	MyCudaMalloc((void**)&dev_y, sizeof(float)* k, 2);

	MyCudaCopy(dev_x, x, sizeof(float)*k, cudaMemcpyHostToDevice, 3);
	MyCudaCopy(dev_y, y, sizeof(float)*k, cudaMemcpyHostToDevice, 4);
	MyCudaCopy(dev_k, &k, sizeof(int), cudaMemcpyHostToDevice, 5);


	functionCuda <<< 1,1 >> >(dev_x, dev_y, dev_k);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "test dot launch failed: %s\n", cudaGetErrorString(cudaStatus));

	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching loopOverPoints2!\n", cudaStatus);

	}


	MyCudaFree(dev_k);
	MyCudaFree(dev_x);
	MyCudaFree(dev_y);
	


}


__global__ void functionCuda(float * x, float * y, int * k)
{

	double sum = dot(x, y, k);

	printf("sum = %lf ", sum);
}