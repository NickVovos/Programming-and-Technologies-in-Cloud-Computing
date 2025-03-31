#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 5000000;
    double start_time, end_time;
    int *data = (int *)malloc(N * sizeof(int));

    MPI_Init(&argc, &argv);                
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);  

    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            data[i] = i;
        }

        start_time = MPI_Wtime();
    }


    MPI_Bcast(data, N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double endTime = MPI_Wtime();
        printf("BCast Time needed: %.6f seconds\n", endTime - start_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);


    if (rank == 0)
    {
        start_time = MPI_Wtime();
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(data, N, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }
    }
    else { 
            MPI_Recv(data, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        double endTime = MPI_Wtime();
        printf("Simple forLoop Time needed: %.6f seconds\n", endTime - start_time);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        start_time = MPI_Wtime();
    }


    int currentStep = 1;
    while (currentStep < size)
        {
            if (rank < currentStep && currentStep+rank < size){
               // printf("rank: %d, to %d \n", rank, currentStep+rank );
                MPI_Send(data, N, MPI_INT, currentStep+rank, 0, MPI_COMM_WORLD);
            }

            if (rank >= currentStep && rank < currentStep*2){
               // printf("to: %d, from %d \n", rank - currentStep , rank );
                MPI_Recv(data, N, MPI_INT, rank - currentStep, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            currentStep= currentStep*2;
        }
    
    if (rank == 0) {
        double endTime = MPI_Wtime();
        printf("Smart forLoop Time needed: %.6f seconds\n", endTime - start_time);
    }

    free(data);
    MPI_Finalize();

    return 0;
}
