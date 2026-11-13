#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 10
#define SHOW 10

void mergeSort(int array[], int length);
void merge(int leftArray[], int leftSize, int rightArray[], int rightSize, int array[]);
int isSorted(int array[], int length);

// Declared outside main so big sizes don't crash
int array[ARRAY_SIZE];
int reference[ARRAY_SIZE];

int main (int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arraySize = ARRAY_SIZE;

    if (rank == 0) {
        // Fill the array with random numbers between 0 and arraySize - 1
        for (int i = 0; i < arraySize; i++) {
            array[i] = rand() % arraySize;
            reference[i] = array[i];
        }

        // Before
        for (int i = 0; i < SHOW && i < arraySize; i++) {
            printf("%d ", array[i]);
        }
        printf("\n===================================================\n");
    }

    // Work out how many elements each process gets
    // If it doesn't divide evenly, the first few processes get 1 extra
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int base = arraySize / size;
    int extra = arraySize % size;
    int offset = 0;

    for (int p = 0; p < size; p++) {
        counts[p] = base + (p < extra ? 1 : 0);
        displs[p] = offset;
        offset += counts[p];
    }

    int localSize = counts[rank];
    int *local = malloc(localSize * sizeof(int));

    // Start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Send each process its part of the array
    MPI_Scatterv(array, counts, displs, MPI_INT, local, localSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its own part
    mergeSort(local, localSize);

    // Merge the parts back in pairs until rank 0 has everything
    // e.g. 4 processes: 1 -> 0 and 3 -> 2, then 2 -> 0
    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            int partner = rank + step;
            if (partner < size) {
                // Receive partner's sorted part
                int partnerSize;
                MPI_Recv(&partnerSize, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int *partnerData = malloc(partnerSize * sizeof(int));
                MPI_Recv(partnerData, partnerSize, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Merge both parts into one
                int *merged = malloc((localSize + partnerSize) * sizeof(int));
                merge(local, localSize, partnerData, partnerSize, merged);

                free(local);
                free(partnerData);
                local = merged;
                localSize += partnerSize;
            }
        } else {
            // Send my sorted part to partner, then stop
            int partner = rank - step;
            MPI_Send(&localSize, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            MPI_Send(local, localSize, MPI_INT, partner, 1, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    // End timer
    double end_time = MPI_Wtime();

    if (rank == 0) {
        // After
        for (int i = 0; i < SHOW && i < arraySize; i++) {
            printf("%d ", local[i]);
        }
        printf("\n");

        double seconds = end_time - start_time;
        printf("\nProgram execution time: %.6f s\n", seconds);

        // Verification: result must be sorted and match the sequential sort
        mergeSort(reference, arraySize);

        int pass = isSorted(local, arraySize);
        for (int i = 0; i < arraySize && pass; i++) {
            if (local[i] != reference[i]) {
                pass = 0;
            }
        }
        printf("Verification: %s (N = %d, processes = %d)\n", pass ? "PASS" : "FAIL", arraySize, size);
    }

    free(counts);
    free(displs);
    free(local);

    MPI_Finalize();
    return 0;
}

void mergeSort (int array[], int length) {
    // base case: 0 or 1 elements are already sorted
    if (length <= 1) {
        return;
    }

    int middle = length / 2;
    int leftSize = middle;
    int rightSize = length - middle;

    int *leftArray = malloc(leftSize * sizeof(int));
    int *rightArray = malloc(rightSize * sizeof(int));
    if (leftArray == NULL || rightArray == NULL) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }

    // Copy each half into its own array
    for (int i = 0; i < leftSize; i++) {
        leftArray[i] = array[i];
    }
    for (int j = 0; j < rightSize; j++) {
        rightArray[j] = array[middle + j];
    }

    mergeSort(leftArray, leftSize);
    mergeSort(rightArray, rightSize);
    merge(leftArray, leftSize, rightArray, rightSize, array);

    free(leftArray);
    free(rightArray);
}

void merge (int leftArray[], int leftSize, int rightArray[], int rightSize, int array[]) {
    int i = 0, l = 0, r = 0; // indices

    // Take the smaller front element from either half
    while (l < leftSize && r < rightSize) {
        if (leftArray[l] < rightArray[r]) {
            array[i] = leftArray[l];
            i++;
            l++;
        } else {
            array[i] = rightArray[r];
            i++;
            r++;
        }
    }

    // Copy whatever is left over
    while (l < leftSize) {
        array[i] = leftArray[l];
        i++;
        l++;
    }

    while (r < rightSize) {
        array[i] = rightArray[r];
        i++;
        r++;
    }
}

// Check the array is in ascending order
int isSorted (int array[], int length) {
    for (int i = 0; i < length - 1; i++) {
        if (array[i] > array[i + 1]) {
            return 0;
        }
    }
    return 1;
}