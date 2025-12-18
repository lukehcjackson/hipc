#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //printf("Hello, World! I am process %d of %d\n", rank, size);

    double my_buffer[100];
    // code to fill the buffer with data
    for (int i = 0; i <= 70; i++) {
        my_buffer[i] = (float)i;
    }

    if (rank == 0) {
        // send 50 doubles to rank 1, with the tag 23.
        int dest = 1;
        int tag = 23;
        MPI_Send(my_buffer, 50, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
    } else if (rank == 1) {
        // receive up to 100 doubles from any rank with any tag
        MPI_Status status;
        MPI_Recv(my_buffer, 100, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count); // how many doubles did we actually receive?
        printf("I've received %d doubles from %d, with the tag %d\n", count, status.MPI_SOURCE, status.MPI_TAG);
    }


    MPI_Finalize();
    return 0;
}
