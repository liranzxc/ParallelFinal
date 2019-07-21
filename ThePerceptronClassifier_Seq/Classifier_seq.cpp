#pragma warning(disable:4996)
#include "Classifier_seq.h"
#include <stdio.h>
#include <stdlib.h>

void Scalar(double ** result, double alfa, double * x, int k);
void Add(double ** W, double * tempResult, int k);
void CreateNewWeights(double ** W, int sign, double * x,int group, double alfa, int k);
double dot(double * W, double * x, int k);

struct Point
{
	double * values;
	int group;
};
void Scalar(double ** result, double alfa, double * x, int k)
{
	for (int i = 0; i < k; i++)
	{
		(*result)[i] = x[i] * alfa;
	}
}

void Add(double ** Result , double * X, int k)
{
	for (int i = 0; i < k; i++)
	{
		(*Result)[i] = (*Result)[i] + X[i];
	}
}
void CreateNewWeights(double ** W, int sign, double * x,int group, double alfa , int k)
{
// alfa *(group - sign ) * pts // old

	double * tempResult =(double*)malloc(sizeof(double)*k);

	//float temp = alfa*(group - sign);

	alfa = alfa * sign;
	Scalar(&tempResult, alfa, x,k);
	Add(W, tempResult,k);

	printf("new W = [%lf ,%lf,%lf,%lf] \n", (*W)[0], (*W)[1], (*W)[2], (*W)[3]);



}
double GetNumberofMisLeadPoints(double * W, Point * pts, int  k, int n)
{
	double counter = 0;
	for (int i = 0; i < n; i++) // over points 3.0
	{
		double F_xi = dot(W, pts[i].values, k);

		if (pts[i].group == 1 && F_xi < 0)   // A group ,mislead
		{
			//printf("Point (%f,%f,%f) is not match , F_xi = %f ,and group = %d \n ", pts[i].values[0], pts[i].values[1], pts[i].values[2], F_xi, pts[i].group);
			
			
			counter++;
		}
		else if (pts[i].group == -1 && F_xi >= 0) // B group ,mislead
		{
			//printf("Point (%f,%f,%f) is not match , F_xi = %f ,and group = %d \n ", pts[i].values[0], pts[i].values[1], pts[i].values[2], F_xi, pts[i].group);

			counter++;
			
		}


	}

	return counter / (n*(1.0));
}
double dot(double * W, double * x, int k)
{
	double scalar_multi = 0.0;

	for (int i = 0; i < k; i++)
	{
		scalar_multi += x[i] * W[i];
	}

	return scalar_multi;
}
int main()
{
	
	int n, k, limit;
	double alfa_zero, alfa_max, QC,q;
	char * data_path = "B://cpp//ThePerceptronClassifier_Seq//ThePerceptronClassifier_Seq//data1.txt";
	FILE* file = fopen(data_path, "r");
	if (file == NULL)
	{
		fprintf(stderr, "\nError opening file\n");
		exit(1);
	}

	fscanf(file, "%d %d %lf %lf %d %lf", &n,&k , &alfa_zero,&alfa_max , &limit,&QC);

	printf("n = %d  k= %d alfa0 = %lf alfa_max = %lf limit = %d  QC =%lf \n", n, k, alfa_zero, alfa_max, limit, QC);

	struct Point * pts =(Point*)malloc(sizeof(Point) * n); // create n array of points

	for (int i = 0; i < n; i++)
	{
		pts[i].values =(double*)malloc(sizeof(double)*(k+1)); // (2,20) exmaple

		for (int j = 0; j < k; j++)
		{
			fscanf(file,"%lf",&pts[i].values[j]);

		}
		pts[i].values[k] = 1; // plus 1 in points xi ( 2,3 ,1 )

		fscanf(file, "%d", &pts[i].group);
	}
	
	fclose(file);

	//  we have array of struct of points 


	/*for (int i = 0; i < n; i++)
	{
		printf("points %f %f group = %d \n", pts[i].values[0], pts[i].values[1], pts[i].group);
	}*/

	


	
	for (double alfa = alfa_zero; alfa <= alfa_max; alfa = alfa + alfa_zero) // running over all alfa
	{
		printf("Alfa %lf \n", alfa);

		double * W =(double*)calloc(k+1, sizeof(double)); // 2.0 <0,0>
		//printf("W = %f %f %f \n", W[0], W[1], W[2]);


		int counter_limit = 0;
		int counter_points_classified_properly = 0;
		while (counter_limit < limit && counter_points_classified_properly != n) // continue untill we have more iters or classified not properly yet
		{
			counter_points_classified_properly = 0;


			for (int i = 0; i < n; i++) // over points 3.0
			{
				double F_xi = dot(W, pts[i].values, k+1); // W*X scalar 

				int sign = F_xi >= 0 ? 1 : -1;


				if (pts[i].group != sign  )   // A group ,mislead
				{
					sign = (pts[i].group - sign) / 2;

					CreateNewWeights(&W, sign, pts[i].values, pts[i].group, alfa, k+1);
					 
					break;
				}

				counter_points_classified_properly++;
				
			}

			//printf("W = %f %f \n", W[0], W[1]);
			counter_limit++;
		} // while close

		 q = GetNumberofMisLeadPoints(W, pts, k+1, n);

		if (q <= QC)
		{
			printf("************ results ********* \n");
			printf("q = %lf \n", q);
			printf("alfa = %lf \n", alfa);

			printf("mini value of W : [");
			for (int i = 0; i < k + 1; i++)
			{
				printf("%lf,", W[i]);


			}
			printf("] \n");

			break;

		}
	/*	else
		{
			printf("not good q = %f \n", q);
			printf("not good w %f %f \n", W[0], W[1]);

		}*/

	}


	

	return 1;
}
