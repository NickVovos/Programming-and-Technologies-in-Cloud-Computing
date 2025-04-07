#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#define WALLIS_TERMS 10000

int main(int argc, char **argv)
{
    int rank, size;
    int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    int myPower = rand() % 10 + 1;

    int *allPowers = NULL;
    int *normalizedPowers = NULL;
    int *termCounts = NULL;
    int *displacements = NULL;
    int *allNValues = NULL;

    if (rank == root)
    {
        allPowers = malloc(size * sizeof(int));
        normalizedPowers = malloc(size * sizeof(int));
        termCounts = malloc(size * sizeof(int));
        displacements = malloc(size * sizeof(int));
    }

    // Συλλογή ισχύος από όλους
    MPI_Gather(&myPower, 1, MPI_INT, allPowers, 1, MPI_INT, root, MPI_COMM_WORLD);

    if (rank == root)
    {
        printf("WALLIS_TERMS %d: \n", WALLIS_TERMS);
        int totalPower = 0, assigned = 0;

        for (int i = 0; i < size; i++)
            totalPower += allPowers[i];

        for (int i = 0; i < size; i++)
        {
            normalizedPowers[i] = (int)round((double)allPowers[i] / totalPower * 100);
            termCounts[i] = (int)round((double)normalizedPowers[i] / 100 * WALLIS_TERMS);
            assigned += termCounts[i];
        }

        // Τελευταίος κόμβος παίρνει τα υπόλοιπα
        termCounts[size - 1] += (WALLIS_TERMS - assigned);

        // Υπολογισμός displacements
        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            displacements[i] = offset;
            offset += termCounts[i];
        }

        // Δημιουργία πίνακα με όλους τους n όρους
        allNValues = malloc(WALLIS_TERMS * sizeof(int));
        for (int i = 0; i < WALLIS_TERMS; i++)
        {
            allNValues[i] = i + 1;
        }

        printf("Κατανομή όρων Wallis:\n");
        for (int i = 0; i < size; i++)
        {
            printf("Κόμβος %d: ισχύς=%d, όροι=%d, ξεκινά από n=%d\n", i, allPowers[i], termCounts[i], displacements[i]);
        }
    }

    // Λήψη τοπικού μεγέθους
    int localSize;
    MPI_Scatter(termCounts, 1, MPI_INT, &localSize, 1, MPI_INT, root, MPI_COMM_WORLD);

    // Τοπικός πίνακας n όρων
    int *localN = malloc(localSize * sizeof(int));
    MPI_Scatterv(allNValues, termCounts, displacements, MPI_INT, localN, localSize, MPI_INT, root, MPI_COMM_WORLD);

    // Τοπικός υπολογισμός Wallis
    double localPi = 1.0;
    for (int i = 0; i < localSize; i++)
    {
        double n = (double)localN[i];
        localPi *= (4.0 * n * n) / (4.0 * n * n - 1);
    }

    // Συνένωση με γινόμενο
    double globalPiProduct = 1.0;
    MPI_Reduce(&localPi, &globalPiProduct, 1, MPI_DOUBLE, MPI_PROD, root, MPI_COMM_WORLD);

    if (rank == root)
    {
        double piEstimate = 2.0 * globalPiProduct;
        printf("\nΤελική προσέγγιση του π (με Wallis και %d όρους): %.15f\n", WALLIS_TERMS, piEstimate);

        free(allPowers);
        free(normalizedPowers);
        free(termCounts);
        free(displacements);
        free(allNValues);
    }

    free(localN);

    MPI_Finalize();
    return 0;
}
