#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int size, rank;
    int n = 20;
    int A[] = {41, 467, 334, 500, 169, 724, 478, 358, 962, 464, 705, 145, 281, 827, 961, 491, 995, 942, 827, 436};
    int proc_pow[] = {3, 2, 4, 1}; 
    int totalPower = 0;
    int normalizedPowers[4];
    int termCounts[4];
    int displacements[4];
    int assigned = 0;
    int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {

        for (int i = 0; i < size; i++) {
            totalPower += proc_pow[i];
        }

        for (int i = 0; i < size; i++) {
            normalizedPowers[i] = (int)round((double)proc_pow[i] / totalPower * 100);
            termCounts[i] = (int)round((double)normalizedPowers[i] / 100 * n);
            assigned += termCounts[i];
        }

        termCounts[size - 1] += (n - assigned);


        int offset = 0;
        for (int i = 0; i < size; i++) {
            displacements[i] = offset;
            offset += termCounts[i];
        }

        for (int i = 0; i < size; i++) {
            printf("rank %d: power=%d, termsCount=%d, Starts from offset=%d\n", i, proc_pow[i], termCounts[i], displacements[i]);
        }
    }


    int localSize;
    MPI_Scatter(termCounts, 1, MPI_INT, &localSize, 1, MPI_INT, root, MPI_COMM_WORLD);

    int local_array[termCounts[rank]];
    printf("\n localsize %d \n", localSize);
    MPI_Scatterv(A, termCounts, displacements, MPI_INT, local_array, localSize, MPI_INT, 0, MPI_COMM_WORLD);


    int first = -1, second = -1;

    for (int i = 0; i < localSize; i++) {
        if (local_array[i] > first) {
            second = first;
            first = local_array[i];
        } else if (local_array[i] > second && local_array[i] != first) {
            second = local_array[i];
        }
    }



    printf("local_max %d \n", first);
    printf("local_second_max %d \n \n \n", second);

    int local_maxes[2] = {first, second};
    int all_maxs[2 * size];

    MPI_Gather(local_maxes, 2, MPI_INT, all_maxs, 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int highest = -1, second_highest = -1;

        for (int i = 0; i < 2 * size; i++) {
            int val = all_maxs[i];
            if (val > highest) {
                second_highest = highest;
                highest = val;
            } else if (val > second_highest && val != highest) {
                second_highest = val;
            }
        }

        printf("The second largest element in the array is: %d\n", second_highest);
    }

    MPI_Finalize();
    return 0;
}
