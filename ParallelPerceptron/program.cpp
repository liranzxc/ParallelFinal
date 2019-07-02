

#include <omp.h>
#include "mpi.h"
#include <stdio.h>
#include "util.h"
#include <stdlib.h>
#include <stdarg.h>
#include <stdexcept>

using namespace std;

int readfromfile(char * path, Point ** pts, int * n, int * k, float * alfa_zero, float * alfa_max, int * limit, float * QC);

	int main(int argc, char *argv[])
	{
		int myrank, size;
		MPI_Status status;
		float exit_label = -1;

		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);


		int n, k, limit;
		float alfa_zero, alfa_max, QC, q;
		struct Point * pts;


		if (myrank == MASTER)
		{
			char * path = "B:\\cpp\\ThePerceptronClassifier_Seq\\ThePerceptronClassifier_Seq\\dataLarge.txt";
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

		// allocation points
		if (myrank != MASTER)
		{
			pts = (Point*)malloc(sizeof(Point) * (n)); // create n array of points
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


#pragma region cuda init later


#pragma endregion

		
// number of cores(processes) in each cpu  = assume 4 

		if (myrank == MASTER)
		{

			float indexerwork = alfa_zero;
			int currectwork = 1;
			while (indexerwork < alfa_max) // dynmaic work
			{

				//// send slaves to process

				float tempwork = indexerwork;
				indexerwork = indexerwork + alfa_zero*NUM_PROCESSES;
				if (indexerwork > alfa_max) indexerwork = alfa_max;
				printf("master sending to %d  work => (%f => %f)\n", currectwork,tempwork, indexerwork);
				fflush(NULL);

				MPI_Send(&tempwork, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&indexerwork, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);

				currectwork = ((currectwork + 1) % size);
				if (currectwork == MASTER && indexerwork < alfa_max) // finish one cycle ,so master do job
			    {
					tempwork = indexerwork;
					indexerwork = indexerwork + alfa_zero*NUM_PROCESSES;
					if (indexerwork > alfa_max) indexerwork = alfa_max;
					// do job
					printf("master finish to cycle \n");
					fflush(NULL);
			    	printf("master %d do job on %f - %f \n", myrank, tempwork, indexerwork);
					fflush(NULL);
					currectwork = ((currectwork + 1) % size);
				}
			}

			for (int worker = 0; worker < size; worker++) // terminal all workers
			{
				MPI_Send(&exit_label, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&exit_label, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
			}
		}
		else
		{
			while (true) // slaves get work
			{
				MPI_Recv(&alfa_zero, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&alfa_max, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status);


				if (alfa_zero == exit_label || alfa_max == exit_label)
				{
					break;
				}

				printf("slave %d do job on %f - %f \n", myrank,alfa_zero,alfa_max);
				fflush(NULL);
			}
		}


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


