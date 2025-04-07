#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include "math.h"

#define DATA_SIZE 50

int main(int argc, char **argv)
{
    int nodeId, totalNodes;
    int masterNode = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &nodeId);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNodes);

    srand(time(NULL) + nodeId);
    int localNodeCapacity = rand() % 10 + 1;

    int *rawCapacities = NULL;
    int *normalizedCapacities = NULL;
    float *initialDataArray = NULL;

    if (nodeId == masterNode)
    {
        rawCapacities = (int *)malloc(totalNodes * sizeof(int));
        normalizedCapacities = (int *)malloc(totalNodes * sizeof(int));
        initialDataArray = (float *)malloc(DATA_SIZE * sizeof(float));

        for (int i = 0; i < totalNodes; i++)
        {
            rawCapacities[i] = 0;
            normalizedCapacities[i] = 0;
        }

        for (int i = 0; i < DATA_SIZE; i++)
        {
            initialDataArray[i] = i;
        }
    }

    MPI_Gather(&localNodeCapacity, 1, MPI_INT, rawCapacities, 1, MPI_INT, masterNode, MPI_COMM_WORLD);

    if (nodeId == masterNode)
    {
        printf("Data size %d:\n", DATA_SIZE);
        printf("Collected raw node capacities:\n");
        int totalCapacity = 0;
        for (int i = 0; i < totalNodes; i++)
        {
            printf("Node %2d processing power: %2d\n", i, rawCapacities[i]);
            totalCapacity += rawCapacities[i];
        }

        int normalizedTotal = 0;
        for (int i = 0; i < totalNodes; i++)
        {
            normalizedCapacities[i] = (int)round((double)rawCapacities[i] / totalCapacity * 100);
            normalizedTotal += normalizedCapacities[i];
        }

        printf("\nInitial dataset: ");
        for (int i = 0; i < DATA_SIZE; i++)
        {
            printf("%.2f ", initialDataArray[i]);
        }
        printf("\n");

        free(rawCapacities);
    }

    int *dataDistribution = NULL;
    int *dataOffsets = NULL;
    int totalDistributed = 0;

    if (nodeId == masterNode)
    {
        dataDistribution = (int *)malloc(totalNodes * sizeof(int));
        dataOffsets = (int *)malloc(totalNodes * sizeof(int));
        int currentOffset = 0;

        for (int i = 0; i < totalNodes; i++)
        {
            dataDistribution[i] = (int)round((double)normalizedCapacities[i] / 100 * DATA_SIZE);
            dataOffsets[i] = currentOffset;
            currentOffset += dataDistribution[i];
            totalDistributed += dataDistribution[i];
        }

        dataDistribution[totalNodes - 1] += (DATA_SIZE - totalDistributed);
    }

    int chunkSize = 0;
    MPI_Scatter(dataDistribution, 1, MPI_INT, &chunkSize, 1, MPI_INT, masterNode, MPI_COMM_WORLD);

    float *chunkData = (float *)malloc(chunkSize * sizeof(float));
    MPI_Scatterv(initialDataArray, dataDistribution, dataOffsets, MPI_FLOAT, chunkData, chunkSize, MPI_FLOAT, masterNode, MPI_COMM_WORLD);

    for (int i = 0; i < chunkSize; i++)
    {
        chunkData[i] *= 3.14f;
    }

    printf("\nNode %d processed %d elements\n", nodeId, chunkSize);

    MPI_Gatherv(chunkData, chunkSize, MPI_FLOAT, initialDataArray, dataDistribution, dataOffsets, MPI_FLOAT, masterNode, MPI_COMM_WORLD);

    free(chunkData);

    if (nodeId == masterNode)
    {
        free(dataDistribution);
        free(dataOffsets);

        printf("\nFinal processed dataset: ");
        for (int i = 0; i < DATA_SIZE; i++)
        {
            printf("%.2f ", initialDataArray[i]);
        }
        printf("\n");

        free(normalizedCapacities);
        free(initialDataArray);
    }

    MPI_Finalize();
    return 0;
}
