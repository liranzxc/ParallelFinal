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
		MPI_Status status;
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
			fflush(NULL);
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




		// init cuda constact 
		Point * dev_pts = NULL;
		int * dev_n = NULL;
		int * dev_k = NULL;
		float * dev_values = NULL;

		double * qMin = (double*)malloc(sizeof(double));
		float * wMin = (float*)malloc(sizeof(float)*(k+1));
		float * alfaMin = (float*)malloc(sizeof(float));


		mallocConstastCuda(pts, n, k, &dev_pts, &dev_n, &dev_k, &dev_values);
		


		//	DoJob(0.01, 0.10,dev_pts,dev_values,dev_n,dev_k,n,k,limit,QC,qMin,wMin,alfaMin);
		//	printf("alfa %f , q = %lf , from process %d \n", *alfaMin, *qMin, myrank);
		//	printf("mini value of W : [%f,%f,%f,%f] \n", wMin[0], wMin[1], wMin[2], wMin[3]);


		


		
// number of cores(processes) in each cpu  = assume 4 

		if (myrank == MASTER)
		{

			float * AlfasFromProcess = (float*)malloc(sizeof(float)*size);

			float currectWorkAlfa = alfa_zero;
			int currectwork = 1;

			float miniumOfminimusAlfas;
			float indexProcessofMinimunAlfa = -1;

			double minQFound;
			float * minWFound = (float*)malloc(sizeof(float)*k + 1);


			while (currectWorkAlfa < alfa_max) // dynmaic work
			{
				float startAlfaWorker = currectWorkAlfa;
				currectWorkAlfa = currectWorkAlfa + alfa_zero*NUM_PROCESSES;

				if (currectWorkAlfa > alfa_max) currectWorkAlfa = alfa_max;
				/*printf("master sending to %d  work => (%f => %f)\n", currectwork,tempwork, indexerwork);
				fflush(NULL);*/
				MPI_Send(&startAlfaWorker, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&currectWorkAlfa, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&alfa_zero, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);

				currectwork = ((currectwork + 1) % size);
				if (currectwork == MASTER && currectWorkAlfa < alfa_max) // finish one cycle ,so master do job
			    {
					startAlfaWorker = currectWorkAlfa;
					currectWorkAlfa = currectWorkAlfa + alfa_zero*NUM_PROCESSES;
					if (currectWorkAlfa > alfa_max) currectWorkAlfa = alfa_max;
			  //  	/*printf("master %d do job on %f - %f \n", myrank, tempwork, indexerwork);
					//fflush(NULL);*/

					DoJob(startAlfaWorker, currectWorkAlfa,alfa_zero, dev_pts, dev_values, dev_n, dev_k, n, k, limit, QC, qMin, wMin, alfaMin);

					/*printf("master %d finish job on %f - %f \n", myrank, tempwork, indexerwork);
					printf("mater finish to do job Qmin = %lf and Alfa Min = %f \n ",*qMin, *alfaMin);
					printf("W = [%f,%f,%f,%f] \n ", wMin[0],wMin[1], wMin[2], wMin[3]);*/

				//	fflush(NULL);

					if (*qMin == 2.0)
					{
						AlfasFromProcess[MASTER] = -1;
					}
					else
					{
						AlfasFromProcess[MASTER] = *alfaMin;

					}

					printf("start recive from process \n ");
					fflush(NULL);
					int counterRecive = 1;
					float tempResultAlfa = 0;
					while (counterRecive < size)
					{
						MPI_Recv(&tempResultAlfa, 1, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
						AlfasFromProcess[status.MPI_SOURCE] = tempResultAlfa;
						counterRecive += 1;


						/*printf("receive from process %d alfa %f \n ",status.MPI_SOURCE,tempResultAlfa);
						fflush(NULL);*/

					}

					//receive from all need to find minimum
					miniumOfminimusAlfas = AlfasFromProcess[0];
					for (int i = 1; i < size; i++)
					{
						if (AlfasFromProcess[i] == -1)
						{
							continue;
						}
						else
						{
							if (miniumOfminimusAlfas > AlfasFromProcess[i])
							{
								miniumOfminimusAlfas = AlfasFromProcess[i];
								indexProcessofMinimunAlfa = i;
							}
						}
					}
				//	printf("the minium alfa master detect is %f \n", miniumOfminimusAlfas);
				//	fflush(NULL);

					if (miniumOfminimusAlfas != -1) // we found some alfa
					{ // found some alfa !!

						int statusSuccfully = 200;
						// get all information for process 
						MPI_Send(&statusSuccfully, 1, MPI_INT, indexProcessofMinimunAlfa, 0, MPI_COMM_WORLD);
						MPI_Recv(&minQFound, 1, MPI_DOUBLE, indexProcessofMinimunAlfa, 0, MPI_COMM_WORLD, &status);
						MPI_Recv(minWFound, k+1, MPI_FLOAT, indexProcessofMinimunAlfa, 0, MPI_COMM_WORLD, &status);

						// need to terminal all process
						break;


					}

					//do job work
					currectwork = ((currectwork + 1) % size);
				}
			}


			if (miniumOfminimusAlfas != -1)
			{

				printf("Found results : \n");
				fflush(NULL);

				printf("alfa %f , q = %lf , from process %d \n", miniumOfminimusAlfas, minQFound, myrank);
				fflush(NULL);

				printf("mini value of W : [");
				for (int i = 0; i < k + 1; i++)
				{
					printf("%f,", minWFound[i]);
					fflush(NULL);


				}
				printf("] \n");
				fflush(NULL);


			}
			else
			{
				printf("Alfa not found !");
				fflush(NULL);

			}

			for (int worker = 0; worker < size; worker++) // terminal all workers
			{
				MPI_Send(&exit_label, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&exit_label, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&exit_label, 1, MPI_FLOAT, currectwork, 0, MPI_COMM_WORLD);

			}
		} 
		
		
		// worker section 
		else // workers
		{
			float step;
			int statusContinue = 0;
			while (true) // slaves get work
			{
				MPI_Recv(&alfa_zero, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&alfa_max, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&step, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, &status);


				if (alfa_zero == exit_label || alfa_max == exit_label || step == exit_label)
				{
					break;
				}

			/*	printf("slave %d do job on %f - %f \n", myrank,alfa_zero,alfa_max);
				fflush(NULL);*/


				DoJob(alfa_zero, alfa_max,step, dev_pts, dev_values, dev_n, dev_k, n, k, limit, QC, qMin, wMin, alfaMin);
			//	printf("slave finish do job  on %f - %f \n", alfa_zero, alfa_max);
				/*printf("slave finish to do job Qmin = %lf and Alfa Min = %f \n ", *qMin, *alfaMin);
				printf("W = [%f,%f,%f,%f] \n ", wMin[0], wMin[1], wMin[2], wMin[3]);
				fflush(NULL);*/

				// return alfaMin , wait for status 

				if (*qMin == 2.0)
				{
					// not found
					float ALFA_NOT_FOUND = -1;
					MPI_Send(&ALFA_NOT_FOUND, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);

				}
				else
				{
					MPI_Send(alfaMin, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);

					MPI_Recv(&statusContinue, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status); // wait for status

					if (statusContinue == 200) // i am the minium  !
					{
						MPI_Send(qMin, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
						MPI_Send(wMin, k+1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);

					}
					else if (statusContinue == exit_label) // some process have a minimun alfa then I
					{
						break;
					}

				}



				//do job work
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
		fflush(NULL);

		printf("n = %d  k= %d alfa0 = %f alfa_max = %f limit = %d  QC =%f \n", *n, *k, *alfa_zero, *alfa_max, *limit, *QC);
		fflush(NULL);

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
		fflush(NULL);

		return 1;

	}


	void FreeConstanstCuda(Point * dev_pts, float * dev_values, int * dev_n, int * dev_k)
	{
		MyCudaFree(dev_pts,03);
		MyCudaFree(dev_values, 04);
		MyCudaFree(dev_n, 05);
		MyCudaFree(dev_k, 06);

	}

	void mallocConstastCuda(Point * pts, int n, int k, Point ** dev_pts, int ** dev_n, int ** dev_k,float ** dev_values)
	{

		MyCudaMalloc((void**)&(*dev_pts), sizeof(Point)* n, 1);
		MyCudaMalloc((void**)&(*dev_values), sizeof(float)* (n*(k + 1)), 2); // value n * (k+1) each point have k+1 dims values

		MyCudaCopy((*dev_pts), pts, sizeof(Point)*n, cudaMemcpyHostToDevice, 4);

		for (int i = 0; i < n; i++)
		{
			MyCudaCopy(&(*dev_values)[i*(k + 1)], &pts[i].values[0], sizeof(float)*(k + 1), cudaMemcpyHostToDevice, 5);

		}


		MyCudaMalloc((void**)&(*dev_n), sizeof(int), 265);
		MyCudaMalloc((void**)&(*dev_k), sizeof(int), 3);
		MyCudaCopy((*dev_n), &n, 1, cudaMemcpyHostToDevice, 5);
		MyCudaCopy((*dev_k), &k, 1, cudaMemcpyHostToDevice, 6);


	}

	//doJob function , each process will execute that function to calcaulate the alfa's.
	// the function will return a minium W , and minium q ,miniumalfa
	// if all alfas dont good enough for QC, return q = 2 
	void DoJob(float alfaZero, float alfaMax,float stepAlfa, Point * dev_pts,float * dev_values,int * dev_n,int * dev_k, int n, int k, int limit,
		float QC, double * qMin,float * wMin,float *alfaMin)
	{
		double maxIteraction = (alfaMax - alfaZero) / stepAlfa;
		int numofSteps = (int)maxIteraction;

		printf(" JOB %f - %f \n ", alfaZero, alfaMax);

		printf("numbers of steps : %d \n", numofSteps);
		fflush(NULL);


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
		

		//copy to up level
		*qMin = minQ;
		*alfaMin = alfaDynamic;
		for (int i = 0; i < k+1; i++)
		{
			wMin[i] = WSaved[indexer][i];
		}

		// free resources
		for (int i = 0; i < numofSteps; i++)
		{
			free(WSaved[i]);
		}
		free(Q_saved);
		free(WSaved);

		//if (minQ != 2)
		//{
		//	//FreeConstanstCuda(dev_pts, dev_values, dev_n, dev_k);
		//}

	}