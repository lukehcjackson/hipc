#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>

#define MAX_RAND 1

double random_number() {
    // returns a random floating point number between 0.0 and MAX_RAND
    return fmod(rand() * ((double) rand() / RAND_MAX), MAX_RAND); 
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(NULL));

    //read start time
    double t0 = MPI_Wtime();

    // read N from the first command line argument
    int N = atoi(argv[1]);

    int total = 0;
    int in = 0;

    int local_in = 0;
    int local_total = 0;

    int rounds = 0;
    int max_rounds = 100;
    float current_pi = 0;
    float tolerance = 1e-7;



    while ((fabs(current_pi - M_PI) > tolerance) && (rounds < max_rounds)) {

        rounds++;

        for (int i = 0; i < N; i++) {

            double x = random_number();
            double y = random_number();

            if ((x*x + y*y) <= 1.0) {
                local_in++;
            }

            local_total++;
            
        }

        float local_pi = ((float)in / (float)total)*4.0;
        printf("I am rank %d on round %d and I think pi is %f \n", rank, rounds, local_pi);

        MPI_Allreduce(&local_in, &in, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_total, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        current_pi = ((float)in / (float)total)*4.0;

        printf("Round: %d Pi: %f \n" , rounds, current_pi);
    }


    

    double t1 = MPI_Wtime(); 
    double total_time = t1 - t0;

    printf("time taken: %fs \n", total_time);

    MPI_Finalize();
    return 0;
}