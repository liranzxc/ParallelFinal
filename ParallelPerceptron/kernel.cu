
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

#include <stdio.h>
#include "util.h"
#include <math.h>
#include <omp.h>
#include <stdarg.h>


__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}


void MyCudaMalloc(void** dev_pointer, size_t size, int error_label)
{
	cudaError_t cudaStatus;

	// points malloc n dims  .
	cudaStatus = cudaMalloc(dev_pointer, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! error_label : %d ", error_label);

		MyCudaFree(*dev_pointer);
	}


}

void MyCudaCopy(void* dest, void * src, size_t size, cudaMemcpyKind kind, int error_label)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dest, src, size, kind);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! error_label : %d", error_label);
	}


}

void MyCudaFree(void * object)
{
	cudaFree(object);
}

__device__ float dot(float * dev_w, float * dev_x,int indexValues, int * dev_k)
{

	float sum = 0;
	for (int i = 0; i < *dev_k + 1; i++)
	{
		sum += dev_w[i] * dev_x[i+ indexValues];

	}

	return sum;

}

//(dev_alfa, dev_values, dev_index_values, dev_W);
__global__ void createNewWeight(float * dev_alfa, float *dev_values,int * indexerValues, float * W_dev)
{
	int i = threadIdx.x;
	W_dev[i] = (*dev_alfa)*dev_values[*indexerValues + i] + W_dev[i];

}



__global__ void	getMisLeadArrayFromPoints(Point * dev_pts, float* dev_values ,float * dev_W, int * dev_mislead, int * dev_k,int * dev_n) {


	int i = blockIdx.x * 1000 + threadIdx.x;

	if (i < *dev_n)
	{
		
	//	printf("working on point[%d] = group = %d \n ", i,dev_pts[i].group);

		int indexValues = i *(*dev_k + 1);

	//	printf("values of points[%d] are => (%f,%f,%f,%f) \n", i, dev_values[indexValues],
		//	dev_values[indexValues+1], dev_values[indexValues+2], dev_values[indexValues + 3]);


		// calaculate fx 

		float fx = dot(dev_W, dev_values,indexValues, dev_k);

//		printf("dot for point[%d] = %f \n", i, fx);

		int sign = fx >= 0 ? 1 : -1;

		if (dev_pts[i].group != sign)   // A group ,mislead
		{
			sign = (dev_pts[i].group - sign) / 2;

			dev_mislead[i] = sign;

		}
		else
		{
			dev_mislead[i] = 0;

		}


	}
}

double ProcessAlfa(Point * dev_pts,float* dev_values, float  * alfa, int *dev_n
	, int *dev_k, int limit, float QC, int n, int k,float ** WSaved)
{
	*WSaved = (float*)malloc((k + 1) * sizeof(float)); // W k+1 dims 
	int * tempresult = (int*)malloc(n * sizeof(int)); // 
	int * mislead = (int*)malloc(n * sizeof(int)); // array of n points , mislead points will be 1 or -1 ,currect=0


	int * dev_mislead = NULL;
	float * dev_W = NULL;
	float * dev_alfa = NULL;
	int * dev_tempresult = NULL;
	cudaError_t cudaStatus;

#pragma region malloc and copy values to GPU


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		FreeFunction(dev_W, dev_alfa, dev_mislead, dev_tempresult);

		return -2;
	}

	// w , mislead_pts , dev_alfa,tempresult,


	MyCudaMalloc((void**)&dev_W, sizeof(float)* (k + 1), 7);
	cudaMemset(dev_W, 0, sizeof(float)* (k + 1));

	//MyCudaCopy(dev_W, W, sizeof(float)*(k + 1), cudaMemcpyHostToDevice, 8);

	MyCudaMalloc((void**)&dev_mislead, sizeof(int)* (n), 9);
	cudaMemset(dev_mislead, 0, sizeof(int)* (n));

	//MyCudaCopy(dev_mislead, mislead, sizeof(int)*(n), cudaMemcpyHostToDevice, 10);

	MyCudaMalloc((void**)&dev_alfa, sizeof(float), 11);
	MyCudaCopy(dev_alfa, alfa, sizeof(float), cudaMemcpyHostToDevice, 12);

	MyCudaMalloc((void**)&dev_tempresult, sizeof(int)*n, 13);
	MyCudaCopy(dev_tempresult, tempresult, sizeof(int)*n, cudaMemcpyHostToDevice, 14);
	cudaMemset(dev_tempresult, 0, sizeof(int)* (n));

	MyCudaCopy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice, 14);




	//// TODO get all values from devices;
	//Point * pts2 = (Point*)malloc(sizeof(Point)*n);


	//MyCudaCopy(W,dev_W, sizeof(float)*(k + 1), cudaMemcpyDeviceToHost, 100);
	//MyCudaCopy(mislead, dev_mislead, sizeof(int)*(n), cudaMemcpyDeviceToHost, 101);
	//MyCudaCopy(alfa, dev_alfa, sizeof(float), cudaMemcpyDeviceToHost, 120);
	//MyCudaCopy(tempresult, dev_tempresult, sizeof(int)*n, cudaMemcpyDeviceToHost, 140);

//MyCudaCopy(pts2, dev_pts, sizeof(int)*n, cudaMemcpyDeviceToHost, 145);

	//MyCudaCopy(&k, dev_k, sizeof(int), cudaMemcpyDeviceToHost, 14);
	//
	//	printf("i=%d0 , cuda w : %lf \n",0, W[0]);
	//	printf(" cuda alfa : %lf \n", *alfa);
	//	printf("i=%d,cuda mislead : %d \n",0, mislead[0]);
	//	printf("i=%d,cuda tempresult : %d \n",0, tempresult[0]); // WORKS
	//printf("cuda n = %d  \n", n);
	//

	//printf("point 0  = (%f,%f,%f,%f)  group = %d \n", pts2[3].values[0], pts2[3].values[1], pts2[3].values[2] ,pts2[3].values[3],pts2[3].group);




#pragma  endregion


	//printf("start computing  Process alfa %f \n",*alfa);
	int threadDims = 1000;
	int blockDims = (n / threadDims) + 1;
	int counter_limit = 0;

	while (counter_limit < limit)
	{

		getMisLeadArrayFromPoints << <blockDims, threadDims >> > (dev_pts, dev_values, dev_W, dev_mislead, dev_k, dev_n);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "loopOverPoints2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
			FreeFunction(dev_W, dev_alfa, dev_mislead, dev_tempresult);
			return -2;

		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching loopOverPoints! : %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
			FreeFunction(dev_W, dev_alfa, dev_mislead, dev_tempresult);
			return -2;

		}

		MyCudaCopy(mislead, dev_mislead, (n) * sizeof(int), cudaMemcpyDeviceToHost, 15);


		int indexerMiss = 0;
		int result;
		// we get a array of mislead values (0 == ok , else (1,-1) false));
		for (indexerMiss = 0; indexerMiss < n; indexerMiss++)
		{
			result = mislead[indexerMiss];
			if (result == -1 || result == 1) // found point that mislead
				break;
		}

		if (indexerMiss == n)
		{
			// all point in good places
			break;
		}
		else
		{
			// need to create a new W 

			*alfa = *alfa*mislead[indexerMiss]; // alfa * sign
			MyCudaCopy(dev_alfa, alfa, sizeof(float), cudaMemcpyHostToDevice, 77);

			*alfa = fabs(*alfa); // back to postive alfa
			
			int indexValues = indexerMiss * (k + 1);
			int * dev_index_values = NULL;
			MyCudaMalloc((void**)&dev_index_values, sizeof(int), 88);
			MyCudaCopy(dev_index_values, &indexValues, sizeof(int), cudaMemcpyHostToDevice, 99);
			createNewWeight << <1, k + 1 >> > (dev_alfa, dev_values, dev_index_values, dev_W);

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "createNewWeight launch failed: %s\n", cudaGetErrorString(cudaStatus));
				FreeFunction(dev_W, dev_alfa, dev_mislead, dev_tempresult);
				return -2;

			}
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching createNewWeight! : %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
				FreeFunction(dev_W, dev_alfa, dev_mislead, dev_tempresult);
				return -2;

			}


			//MyCudaCopy(W, dev_W, sizeof(float)*(k + 1), cudaMemcpyDeviceToHost, 70);

			//printf("new W = [%f ,%f,%f,%f] \n", (W)[0], (W)[1], (W)[2], (W)[3]);


		}


		counter_limit++;
	}

	// need to calcate the q , get all mislead point 



	getMisLeadArrayFromPoints << <blockDims, threadDims >> > (dev_pts, dev_values, dev_W, dev_tempresult, dev_k, dev_n);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "getMisLeadArrayFromPoints Second Time launch failed: %s\n", cudaGetErrorString(cudaStatus));
		FreeFunction(dev_W, dev_alfa, dev_mislead, dev_tempresult);
		return -2;

	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getMisLeadArrayFromPoints Second Time! : %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
		FreeFunction(dev_W, dev_alfa, dev_mislead, dev_tempresult);
		return -2;

	}

	MyCudaCopy(tempresult, dev_tempresult, (n) * sizeof(int), cudaMemcpyDeviceToHost, 15);

	int sumOFmisLead = 0;
	//#pragma omp parallel for reduction(+:sumOFmisLead)
		for (int i = 0; i < n; i++)
		{
			if (tempresult[i] != 0)
			{
				sumOFmisLead += 1;
			}
		}

		//cudaMemset(dev_W, 0, sizeof(float)* (k + 1)); // clean up
		//cudaMemset(dev_mislead, 0, sizeof(int)* (n));
		//cudaMemset(dev_tempresult, 0, sizeof(int)* (n));


		double q = sumOFmisLead / (n*(1.0));

		MyCudaCopy(*WSaved, dev_W, sizeof(float)*(k + 1), cudaMemcpyDeviceToHost, 70); // copy W


		FreeFunction(dev_W, dev_alfa, dev_mislead, dev_tempresult);

		if (q <= QC)
				return q;
		else
			return 2; // q that never will get and larger from all q possiblies .



}


cudaError_t FreeFunction(float * dev_W ,float * dev_alfa, int * dev_mislead ,int * dev_tempresult)
{
	cudaError_t cudaStatus;


		cudaStatus = cudaFree(dev_W);
		if (cudaStatus != cudaSuccess) {

			printf("failed to free cuda - W  \n");
		}
		cudaStatus = cudaFree(dev_mislead);
		if (cudaStatus != cudaSuccess) {

			printf("failed to free cuda - mislead points \n");
		}
	
		cudaStatus = cudaFree(dev_tempresult);
		if (cudaStatus != cudaSuccess) {
			
			printf("failed to free cuda - tempresult \n");
		}

		cudaStatus = cudaFree(dev_alfa);
		if (cudaStatus != cudaSuccess) {

			printf("failed to free cuda - alfa \n");
		}
	
	return cudaStatus;
}

