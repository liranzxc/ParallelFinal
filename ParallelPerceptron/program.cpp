#define _CRT_SECURE_NO_WARNINGS


#include <omp.h>
#include "mpi.h"
#include <stdio.h>
#include "util.h"
#include <stdlib.h>
#include <stdarg.h>
#include <stdexcept>
#include <math.h>
using namespace std;

int readfromfile(char * path, Point ** pts, int * n, int * k, float * alfa_zero, float * alfa_max, int * limit, float * QC);

	int main(int argc, char *argv[])
	{
		int myrank, size;
		//MPI_Status status;
		float exit_label = -1;

		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);


		int n, k, limit;
		float alfa_zero, alfa_max, QC;
		struct Point * pts;


		if (myrank == MASTER)
		{
			char * path = "B:\\cpp\\ThePerceptronClassifier_Seq\\ThePerceptronClassifier_Seq\\dataset.txt";
			if (readfromfile(path, &pts, &n, &k, &alfa_zero, &alfa_max, &limit, &QC) == 0) {
				printf("LIRAN ERROR : error reading from file \n");
				MPI_Abort(MPI_COMM_WORLD, 1);
				exit(1);
			}

			printf("master finish reading from file \n");
		}

		// bcast all information
		MPI_Bcast(&k, 1, MPI_INT, MASTER, MPI_COMM_WORLD); // for dim type point
		MPI_Bcast(&limit, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&QC, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

#pragma region master_boardcast status - all processes have same data without alfas

	//	 allocation points
		if (myrank != MASTER)
		{
			pts = (Point*)malloc(sizeof(Point) * (n)); // create n array of points
			#pragma omp parallel for
			for (int i = 0; i < n; i++)
			{
				pts[i].values = (float*)malloc(sizeof(float)*(k + 1)); // (2,20) exmaple
			}
		}
	

		//boardcast all values
		for (int i = 0; i < n; i++)
		{
			MPI_Bcast(&pts[i].values[0], k + 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&pts[i].group, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		}


		

#pragma  endregion 



		if (myrank == MASTER || myrank == 1)
		{
			DoJob(0.01, 0.04, pts, n, k, limit, QC, myrank);

		}


		
// number of cores(processes) in each cpu  = assume 4 

		//if (myrank == MASTER)
		//{

		//	float indexerwork = alfa_zero;
		//	int currectwork = 1;
		//	while (indexerwork < alfa_max) // dynmaic work
		//	{

		//		//// send slaves to process

		//		float tempwork = indexerwork;
		//		indexerwork = indexerwork + alfa_zero*NUM_PROCESSES;
		//		if (indexerwork > alfa_max) indexerwork = alfa_max;
		//		printf("master sending to %d  work => (%f => %f)\n", currectwork,tempwork, indexerwork);
		//		fflush(NULL);

		//		MPI_Send(&tempwork, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
		//		MPI_Send(&indexerwork, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);

		//		currectwork = ((currectwork + 1) % size);
		//		if (currectwork == MASTER && indexerwork < alfa_max) // finish one cycle ,so master do job
		//	    {
		//			tempwork = indexerwork;
		//			indexerwork = indexerwork + alfa_zero*NUM_PROCESSES;
		//			if (indexerwork > alfa_max) indexerwork = alfa_max;
		//			// do job
		//			printf("master finish to cycle \n");
		//			fflush(NULL);
		//	    	printf("master %d do job on %f - %f \n", myrank, tempwork, indexerwork);



		//			fflush(NULL);
		//			currectwork = ((currectwork + 1) % size);
		//		}
		//	}

		//	for (int worker = 0; worker < size; worker++) // terminal all workers
		//	{
		//		MPI_Send(&exit_label, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
		//		MPI_Send(&exit_label, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
		//	}
		//}
		//else
		//{
		//	while (true) // slaves get work
		//	{
		//		MPI_Recv(&alfa_zero, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status);
		//		MPI_Recv(&alfa_max, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status);


		//		if (alfa_zero == exit_label || alfa_max == exit_label)
		//		{
		//			break;
		//		}

		//		printf("slave %d do job on %f - %f \n", myrank,alfa_zero,alfa_max);
		//		fflush(NULL);
		//	}
		//}


		MPI_Finalize();

		return 0;

	}

	int readfromfile(char * path, Point ** pts, int * n, int * k, float * alfa_zero, float * alfa_max, int * limit, float * QC)
	{

		FILE* file = fopen(path, "r");
		if (file == NULL)
		{
			fprintf(stderr, "\nError opening file\n");
			return 0;
		}

		fscanf(file, "%d %d %f %f %d %f", n, k, alfa_zero, alfa_max, limit, QC);

		printf("n = %d  k= %d alfa0 = %f alfa_max = %f limit = %d  QC =%f \n", *n, *k, *alfa_zero, *alfa_max, *limit, *QC);

		*pts = (Point*)malloc(sizeof(Point) * (*n)); // create n array of points

		for (int i = 0; i < *n; i++)
		{
			(*pts)[i].values = (float*)malloc(sizeof(float)*(*k + 1)); // (2,20) exmaple

			for (int j = 0; j < *k; j++)
			{
				fscanf(file, "%f", &(*pts)[i].values[j]);

			}
			(*pts)[i].values[*k] = 1; // plus 1 in points xi ( 2,3 ,1 )

			fscanf(file, "%d", &(*pts)[i].group);
		}

		fclose(file);

		printf("finish reading from file \n");

		return 1;

	}




	void DoJob(float alfaZero, float alfaMax, Point * pts, int n, int k, int limit, float QC, int myrank)
	{

		Point * dev_pts = NULL;
		int * dev_n = NULL;
		int * dev_k = NULL;

		float * dev_values = NULL;

		// points , n , k 
		
		MyCudaMalloc((void**)&dev_pts, sizeof(Point)* n, 1);
		MyCudaMalloc((void**)&dev_values, sizeof(float)* (n*(k+1)), 2); // value n * (k+1) each point have k+1 dims values

		MyCudaCopy(dev_pts, pts, sizeof(Point)*n, cudaMemcpyHostToDevice, 4);

		for (int i = 0; i < n; i++)
		{
			MyCudaCopy(&dev_values[i*(k+1)], &pts[i].values[0], sizeof(float)*(k+1), cudaMemcpyHostToDevice, 5);

		}


		MyCudaMalloc((void**)&dev_n, sizeof(int), 265);
		MyCudaMalloc((void**)&dev_k, sizeof(int), 3);
		MyCudaCopy(dev_n, &n,1, cudaMemcpyHostToDevice, 5);
		MyCudaCopy(dev_k, &k, 1, cudaMemcpyHostToDevice, 6);



		// need to waiting 

		double maxIteraction = alfaMax / alfaZero;
		int numofSteps = (int)maxIteraction;

		printf("numbers of steps : %d \n", numofSteps);


		//			DoJob(0.1, 1, pts, n, k, limit, QC, myrank);


		double * Q_saved = (double*)malloc(numofSteps*sizeof(double));
		float ** WSaved = (float**)malloc(numofSteps * sizeof(float * ));


		#pragma omp parallel for
		for (int i = 0; i < numofSteps; i++) // running over all alfa
		{

			float * tempAlfa = (float*)malloc(sizeof(float));
			*tempAlfa = alfaZero + i*alfaZero;
			
			Q_saved[i] = ProcessAlfa(dev_pts, dev_values,tempAlfa, dev_n, dev_k, limit, QC, n , k,&WSaved[i]);
			//printf("alfa %f , q = %lf , from process %d \n", fabsf(*tempAlfa), Q_saved[i], myrank);

		}
	


		//get mininum q from all q's , get W save ,and alfa

		double minQ = Q_saved[0];
		int indexer = 0;

		for (int i = 1; i < numofSteps; i++)
		{
			if (Q_saved[i] < minQ)
			{
				minQ = Q_saved[i];
				indexer = i;
			}
		}

		float alfaDynamic = alfaZero + indexer *alfaZero;
		printf("alfa %f , q = %lf , from process %d \n", alfaDynamic, minQ, myrank);
		printf("mini value of W : [%f,%f,%f,%f] \n", WSaved[indexer][0], WSaved[indexer][1], WSaved[indexer][2], WSaved[indexer][3]);


		// free resources
		for (int i = 0; i < numofSteps; i++)
		{
			free(WSaved[i]);
		}
		free(Q_saved);
		free(WSaved);

		if (minQ != 2)
		{
			cudaFree(dev_pts);
			cudaFree(dev_values);
		}


		// send to master 



		


	}