#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *data;

    if (rank == 0) {
        // allocate memory on rank 0 for 10 * size elements, fill the array with increasing values
        data = malloc(sizeof(double) * 10 * size);
        for (int i = 0; i < 10 * size; i++) data[i] = i;
    }

    double my_data[10];

    // scatter 10 doubles to each process and store it in the my_data array
    //ignored by everything except rank 0 (receiving processes)
    //see                                                      V
    MPI_Scatter(data, 10, MPI_DOUBLE, my_data, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // serialise the output of each processes data using MPI_Barrier for cleanliness
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("I'm rank %d, my data is: ", rank);
            for (int j = 0; j < 10; j++) printf("%lf, ", my_data[j]);
            printf("\n");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    if (rank == size-1) {
        // allocate memory for the gather on rank (size-1)
        data = malloc(sizeof(double) * 10 * size);
    }
    // gather 10 doubled from each process and place them into the data array of rank (size-1)
    //ignored by everything except last rank (the number of which is equal to size-1)
    //see                                                       V
    MPI_Gather(my_data, 10, MPI_DOUBLE, data, 10, MPI_DOUBLE, size-1, MPI_COMM_WORLD);

    if (rank == size-1) {
        printf("I'm the last processor and I've gathered: \n");
        for (int i = 0; i < 10 * size; i++) {
            printf("%lf, ", data[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
}
