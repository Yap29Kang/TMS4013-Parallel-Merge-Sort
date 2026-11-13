/*
 * ============================================================================
 * File:        mpi_mergesort.c
 * Project:     Part II - Parallelism (Parallel Merge Sort using MPI)
 *
 * Description:
 *   Parallel Merge Sort using MPI distributed memory.
 *   Rank 0 generates N random integers, distributes balanced chunks to all
 *   processes with MPI_Scatterv, each process sorts its local chunk, and
 *   sorted chunks are merged back to Rank 0 using a binary-tree merge.
 *
 * Input (Command-Line Arguments):
 *   mpirun -np <P> ./mpi_mergesort [N]
 *     N : Optional number of integers to sort (default: 1000000)
 *
 * Output:
 *   - Sample elements before & after sorting (first 10 elements)
 *   - Load balance summary (min/max elements per process)
 *   - Timing breakdown (Scatter, Local Sort, Tree Merge, Total, Sequential)
 *   - Speedup, Parallel Efficiency, and estimated Memory Usage (MB)
 *   - Verification result (PASS/FAIL against ascending check & sequential sort)
 *
 * Compilation:
 *   mpicc -O2 -Wall -o mpi_mergesort mpi_mergesort.c
 *
 * Execution:
 *   mpirun -np 4 ./mpi_mergesort 10         (small visual test)
 *   mpirun -np 4 ./mpi_mergesort 10000000   (scaling benchmark)
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SHOW 10

void mergeSort(int array[], int length);
void mergeSortRec(int array[], int temp[], int left, int right);
void merge(int array[], int temp[], int left, int middle, int right);
void mergeTwoParts(int leftArray[], int leftSize, int rightArray[], int rightSize, int outArray[]);
int isSorted(int array[], int length);

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Read array size from command line (default: 1,000,000) */
    int arraySize = (argc > 1) ? atoi(argv[1]) : 1000000;
    if (arraySize <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: array size must be > 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* Allocate full input & reference arrays ONLY on Rank 0 to save RAM on workers */
    int *array = NULL;
    int *reference = NULL;

    if (rank == 0) {
        array = (int *)malloc((size_t)arraySize * sizeof(int));
        reference = (int *)malloc((size_t)arraySize * sizeof(int));
        if (array == NULL || reference == NULL) {
            fprintf(stderr, "Rank 0 out of memory for N = %d\n", arraySize);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Fixed seed ensures identical dataset across different process counts */
        srand(42);
        for (int i = 0; i < arraySize; i++) {
            array[i] = rand() % arraySize;
            reference[i] = array[i];
        }

        printf("Before (first %d): ", SHOW < arraySize ? SHOW : arraySize);
        for (int i = 0; i < SHOW && i < arraySize; i++) {
            printf("%d ", array[i]);
        }
        printf("\n===================================================\n");
    }

    /* Load balancing: distribute elements evenly (difference at most 1) */
    int *counts = (int *)malloc((size_t)size * sizeof(int));
    int *displs = (int *)malloc((size_t)size * sizeof(int));
    int base = arraySize / size;
    int extra = arraySize % size;
    int offset = 0;

    for (int p = 0; p < size; p++) {
        counts[p] = base + (p < extra ? 1 : 0);
        displs[p] = offset;
        offset += counts[p];
    }

    int localSize = counts[rank];
    int *local = (int *)malloc((localSize > 0 ? (size_t)localSize : 1) * sizeof(int));
    if (local == NULL) {
        fprintf(stderr, "Rank %d out of memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Start parallel timer */
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /* Step 1: Scatter chunks from Rank 0 to each process */
    MPI_Scatterv(array, counts, displs, MPI_INT,
                 local, localSize, MPI_INT, 0, MPI_COMM_WORLD);
    double after_scatter = MPI_Wtime();

    /* Free Rank 0's input buffer early to reduce peak memory during merge */
    if (rank == 0) {
        free(array);
        array = NULL;
    }

    /* Step 2: Each process sorts its own local chunk */
    double sort_start = MPI_Wtime();
    mergeSort(local, localSize);
    double sort_end = MPI_Wtime();

    /* Step 3: Binary-tree merge until Rank 0 holds the complete sorted array */
    double merge_start = MPI_Wtime();
    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            int partner = rank + step;
            if (partner < size) {
                int partnerSize = 0;
                MPI_Recv(&partnerSize, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int *partnerData = (int *)malloc((partnerSize > 0 ? (size_t)partnerSize : 1) * sizeof(int));
                if (partnerSize > 0) {
                    MPI_Recv(partnerData, partnerSize, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

                int *merged = (int *)malloc(((localSize + partnerSize) > 0 ? (size_t)(localSize + partnerSize) : 1) * sizeof(int));
                mergeTwoParts(local, localSize, partnerData, partnerSize, merged);

                free(local);
                free(partnerData);
                local = merged;
                localSize += partnerSize;
            }
        } else {
            int partner = rank - step;
            MPI_Send(&localSize, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            if (localSize > 0) {
                MPI_Send(local, localSize, MPI_INT, partner, 1, MPI_COMM_WORLD);
            }
            free(local);
            local = NULL;
            break;
        }
        step *= 2;
    }
    double merge_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    /* Collect min/max local sort times across all ranks */
    double local_sort_time = sort_end - sort_start;
    double max_sort_time = 0.0, min_sort_time = 0.0;
    MPI_Reduce(&local_sort_time, &max_sort_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sort_time, &min_sort_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("After  (first %d): ", SHOW < arraySize ? SHOW : arraySize);
        for (int i = 0; i < SHOW && i < arraySize; i++) {
            printf("%d ", local[i]);
        }
        printf("\n");

        double scatter_time = after_scatter - start_time;
        double merge_time   = merge_end - merge_start;
        double total_time   = end_time - start_time;

        /* Run sequential sort on reference array for timing & verification */
        double seq_start = MPI_Wtime();
        mergeSort(reference, arraySize);
        double seq_end = MPI_Wtime();
        double seq_time = seq_end - seq_start;

        int pass = isSorted(local, arraySize);
        for (int i = 0; i < arraySize && pass; i++) {
            if (local[i] != reference[i]) {
                pass = 0;
            }
        }

        double speedup = seq_time / total_time;
        double efficiency = (speedup / size) * 100.0;

        /* Memory estimates: worker holds local + temp (2 * maxChunk), Rank 0 peaks at 2 * N during final merge */
        int minChunk = counts[size - 1];
        int maxChunk = counts[0];
        double workerMemMB = (2.0 * maxChunk * sizeof(int)) / (1024.0 * 1024.0);
        double rootPeakMB  = (2.0 * arraySize * sizeof(int)) / (1024.0 * 1024.0);

        printf("\n================ MPI MERGE SORT RESULTS ================\n");
        printf("Processes (P)       : %d\n", size);
        printf("Array Size (N)      : %d\n", arraySize);
        printf("Elements per Rank   : min = %d, max = %d\n", minChunk, maxChunk);
        printf("Scatter Time        : %.6f s\n", scatter_time);
        printf("Local Sort (min/max): %.6f s / %.6f s\n", min_sort_time, max_sort_time);
        printf("Tree Merge Time     : %.6f s\n", merge_time);
        printf("Total Parallel Time : %.6f s\n", total_time);
        printf("Sequential Time     : %.6f s\n", seq_time);
        printf("Speedup             : %.3fx\n", speedup);
        printf("Parallel Efficiency : %.2f%%\n", efficiency);
        printf("Worker Mem (Est.)   : %.2f MB per rank\n", workerMemMB);
        printf("Rank 0 Peak (Est.)  : %.2f MB (final merge)\n", rootPeakMB);
        printf("Verification        : %s (N = %d, processes = %d)\n",
               pass ? "PASS" : "FAIL", arraySize, size);
        printf("========================================================\n");

        free(reference);
        free(local);
    }

    free(counts);
    free(displs);
    MPI_Finalize();
    return 0;
}

/* Single-allocation Merge Sort wrapper (avoids repeated malloc/free in recursion) */
void mergeSort(int array[], int length) {
    if (length <= 1) {
        return;
    }
    int *temp = (int *)malloc((size_t)length * sizeof(int));
    if (temp == NULL) {
        fprintf(stderr, "Out of memory in mergeSort\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mergeSortRec(array, temp, 0, length - 1);
    free(temp);
}

void mergeSortRec(int array[], int temp[], int left, int right) {
    if (left >= right) {
        return;
    }
    int middle = left + (right - left) / 2;
    mergeSortRec(array, temp, left, middle);
    mergeSortRec(array, temp, middle + 1, right);
    merge(array, temp, left, middle, right);
}

void merge(int array[], int temp[], int left, int middle, int right) {
    int i = left, l = left, r = middle + 1;
    while (l <= middle && r <= right) {
        if (array[l] <= array[r]) {
            temp[i++] = array[l++];
        } else {
            temp[i++] = array[r++];
        }
    }
    while (l <= middle) temp[i++] = array[l++];
    while (r <= right)  temp[i++] = array[r++];
    for (int k = left; k <= right; k++) {
        array[k] = temp[k];
    }
}

/* Merge two separate sorted arrays into outArray (used in MPI tree merge) */
void mergeTwoParts(int leftArray[], int leftSize, int rightArray[], int rightSize, int outArray[]) {
    int i = 0, l = 0, r = 0;
    while (l < leftSize && r < rightSize) {
        if (leftArray[l] <= rightArray[r]) {
            outArray[i++] = leftArray[l++];
        } else {
            outArray[i++] = rightArray[r++];
        }
    }
    while (l < leftSize)  outArray[i++] = leftArray[l++];
    while (r < rightSize) outArray[i++] = rightArray[r++];
}

/* Check that the array is in ascending order */
int isSorted(int array[], int length) {
    for (int i = 0; i < length - 1; i++) {
        if (array[i] > array[i + 1]) {
            return 0;
        }
    }
    return 1;
}