#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

//size of plate
#define M 200
#define N 1000

double **alloc_2d_array(int m, int n) {
  	double **x;
  	int i;

  	x = (double **)malloc(m*sizeof(double *));
  	x[0] = (double *)calloc(m*n,sizeof(double));
  	for ( i = 1; i < m; i++ )
    	x[i] = &x[0][i*n];
	return x;
}

void free_2d_array(double ** array) {
	free(array[0]);
	free(array);
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("Heated Plate calculation\n");

	// arrays for recording temperatures
    //instead of allocating two big M*N arrays, allocate smaller arrays that will go onto each CPU
    //int chunkSize = (N / size) + 2;
    //assert((chunkSize * size) - (size*2) == N);

    //find the indices each chunk of the array will be at
    //j is left to right, i is up and down
    int startj = rank * (N/size) - 1;
    if (startj < 0) startj = 0;

    int endj = (rank+1) * (N/size);
    if (endj >= N-1) endj = N-1;

    //find size of each chunk
    int chunkSize = endj - startj + 1;

    if (rank == 0) printf("Rank %d start and end: %d, %d, requires (M x N): (%d x %d)\n", rank, startj, endj, M, chunkSize);
    

	double** u = alloc_2d_array(M, chunkSize);	
	double** w = alloc_2d_array(M, chunkSize);	

	double diff;
	double epsilon = 0.00001;
	int iterations;
	int iterations_print;
	double mean;

    //create custom datatype for M doubles in one column
    //the gap between them is chunkSize - i.e. the size of one row on this processor
    MPI_Datatype my_column;
    MPI_Type_vector(M, 1, chunkSize, MPI_DOUBLE, &my_column);
    MPI_Type_commit(&my_column);
  
	if (rank == 0) printf("  Spatial grid of %d by %d points.\n", M, chunkSize);
	if (rank == 0) printf("  The iteration will be repeated until the change is <= %lf\n", epsilon); 

    // Set the boundary values, which don't change.
	mean = 0.0;

	for (int i = 1; i < M-1; i++) {
        if (rank == 0) {
            w[i][0] = 100.0;
        }
		if (rank == size-1) {
            w[i][chunkSize-1] = 100.0;
        } 
	}

	for (int j = 0; j < chunkSize; j++) {
        w[M-1][j] = 100.0;
        w[0][j] = 0.0;
    }

    // Average the boundary values, to come up with a reasonable initial value for the interior.
    double local_mean = 0.0; 
	for (int i = 1; i < M-1; i++) {		
        //mean += w[i][0] + w[i][N-1];

        if (rank == 0) {
            local_mean += w[i][0];
        }
        if (rank == size-1) {
            local_mean += w[i][chunkSize-1];
        }

	}

    //the first and last chunk have one ghost column
    //each chunk has two ghost columns
    //these columns will have had their values set above but we don't want to include them
    //in this average

    //on everything except rank 0 and rank N-1, start from index 1 and go to index chunkSize-1
    int mystart = (rank == 0) ? 0 : 1;
    int myend = (rank == size-1) ? chunkSize : chunkSize-1;
    for (int j = mystart; j < myend; j++) {
        local_mean += w[M-1][j] + w[0][j];
    }

    //could also do a in-place reduction and take out local_mean altogether
    MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //all cpus now have the right sum of local means, so divide by # of cells

	mean = mean / (double) ( 2 * M + 2 * N - 4 );
	printf("\n MEAN = %lf\n", mean);

    // Initialize the interior solution to the mean value. 
	for (int i = 1; i < M - 1; i++) {
		for (int j = 1; j < chunkSize - 1; j++) {
    	 		w[i][j] = mean;
		}
	}

    // iterate until the new solution W differs from the old solution U by no more than EPSILON. 
	iterations = 0;
	iterations_print = 1000; // print an update every 1000 iterations

	diff = epsilon;

	while (epsilon <= diff) {

        //communicate between processes

        //calculate the rank of the process to the left and right
        //using MPI_PROC_NULL if this is the first or last process
        int left = (rank - 1) < 0 ? MPI_PROC_NULL : rank - 1;
        int right = (rank + 1) >= size ? MPI_PROC_NULL : rank + 1;

        //exchange column 1 with ghost column chunkSize-1
        MPI_Sendrecv(&(w[0][1]), 1, my_column, left, 0, 
                     &(w[0][chunkSize-1]), 1, my_column, right, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //exchange column chunkSize-2 with ghost column 0
        MPI_Sendrecv(&(w[0][chunkSize-2]), 1, my_column, right, 0,
                     &(w[0][0]), 1, my_column, left, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        // Save the old solution in U. 
		for (int i = 0; i < M; i++) {
     		for (int j = 0; j < chunkSize; j++) {
        		u[i][j] = w[i][j];
        	}
      	}

        // Determine the new estimate of the solution at the interior points. 
        // The new solution W is the average of north, south, east and west neighbors.
      	for (int i = 1; i < M - 1; i++) {
        	for (int j = 1; j < chunkSize - 1; j++) {
        		w[i][j] = (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) / 4.0;
        	}
      	}

        // Find the largest difference between the old and new values
		diff = 0.0;
		for (int i = 1; i < M - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
		     	if (diff < fabs(w[i][j]-u[i][j])) {
	          		diff = fabs(w[i][j]-u[i][j]);
     			}
			}
		}

        //find the maximum difference across all processors
        MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		iterations++;
		if (iterations % iterations_print == 0) {
			if (rank == 0) printf("  %8d  %f\n", iterations, diff);
		}
	}

	if (rank == 0) printf("\n  %8d  %f\n", iterations, diff);

	if (rank == 0) printf("\n  End of execution.\n");

    // write output to a csv file
    /*
    FILE *output = fopen("./output.csv", "w");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N-1; j++) {
            fprintf(output, "%lf,", w[i][j]);
        }
        fprintf(output, "%lf\n", w[i][N-1]);
    }
    fclose(output);
    */

    free_2d_array(w);
    free_2d_array(u);

    MPI_Type_free(&my_column);
    MPI_Finalize();
    return 0;
}
