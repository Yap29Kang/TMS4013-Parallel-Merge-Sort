/*
 * ============================================================================
 * File:        openmp_mergesort.c
 * Project:     Part III - Parallelism (Parallel Merge Sort using OpenMP)
 *
 * Description:
 *   Parallel Merge Sort using OpenMP shared memory multithreading.
 *   Follows the exact same divide-and-conquer algorithm as mpi_mergesort.c:
 *   - Master thread generates N random integers with fixed seed (42).
 *   - Balanced chunks are allocated to T threads (base + remainder distribution).
 *   - Each thread independently sorts its local subarray using merge sort.
 *   - Sorted subarrays are merged in parallel using a binary-tree merge.
 *
 * Input (Command-Line Arguments):
 *   ./openmp_mergesort [N] [T]
 *     N : Optional number of integers to sort (default: 1000000)
 *     T : Optional number of OpenMP threads (default: omp_get_max_threads())
 *
 * Output:
 *   - Sample elements before & after sorting (first 10 elements)
 *   - Load balance summary (min/max elements per thread)
 *   - Timing breakdown (Partition, Local Sort min/max, Tree Merge, Total, Sequential)
 *   - Speedup, Parallel Efficiency, and estimated Memory Usage (MiB)
 *   - Verification result (PASS/FAIL against ascending check & sequential sort)
 *
 * Compilation:
 *   gcc -O2 -Wall -fopenmp -o openmp_mergesort openmp_mergesort.c
 *
 * Execution:
 *   ./openmp_mergesort 10 4          (small visual test)
 *   ./openmp_mergesort 10000000 4    (scaling benchmark)
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define SHOW 10

void mergeSort(int array[], int length);
void mergeSortRec(int array[], int temp[], int left, int right);
void merge(int array[], int temp[], int left, int middle, int right);
void mergeTwoParts(int leftArray[], int leftSize, int rightArray[], int rightSize, int outArray[]);
int isSorted(int array[], int length);

int main(int argc, char *argv[]) {
    /* Read array size from command line (default: 1,000,000) */
    int arraySize = (argc > 1) ? atoi(argv[1]) : 1000000;
    if (arraySize <= 0) {
        fprintf(stderr, "Error: array size must be > 0\n");
        return 1;
    }

    /* Read thread count from command line (default: max available threads) */
    int numThreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    if (numThreads <= 0) {
        numThreads = 1;
    }
    omp_set_num_threads(numThreads);

    /* Allocate full input array, temporary scratch buffer for tree merge, and reference */
    int *array = (int *)malloc((size_t)arraySize * sizeof(int));
    int *tempMerge = (int *)malloc((size_t)arraySize * sizeof(int));
    int *reference = (int *)malloc((size_t)arraySize * sizeof(int));

    if (array == NULL || tempMerge == NULL || reference == NULL) {
        fprintf(stderr, "Error: Out of memory allocating arrays for N = %d\n", arraySize);
        free(array);
        free(tempMerge);
        free(reference);
        return 1;
    }

    /* Fixed seed ensures identical dataset across different thread counts & MPI comparison */
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

    /* Load balancing: distribute elements evenly across threads (difference at most 1) */
    int *counts = (int *)malloc((size_t)numThreads * sizeof(int));
    int *displs = (int *)malloc((size_t)numThreads * sizeof(int));
    int *curSizes = (int *)malloc((size_t)numThreads * sizeof(int));
    double *localSortTimes = (double *)malloc((size_t)numThreads * sizeof(double));

    if (counts == NULL || displs == NULL || curSizes == NULL || localSortTimes == NULL) {
        fprintf(stderr, "Error: Out of memory allocating thread metadata\n");
        return 1;
    }

    int base = arraySize / numThreads;
    int extra = arraySize % numThreads;
    int offset = 0;

    for (int t = 0; t < numThreads; t++) {
        counts[t] = base + (t < extra ? 1 : 0);
        displs[t] = offset;
        curSizes[t] = counts[t];
        localSortTimes[t] = 0.0;
        offset += counts[t];
    }

    /* Start parallel timer */
    double start_time = omp_get_wtime();

    /* Step 1: Thread chunk mapping / partition (in shared memory this is instantaneous) */
    double after_partition = omp_get_wtime();

    /* Step 2: Each thread independently sorts its local contiguous chunk */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double t0 = omp_get_wtime();
        if (counts[tid] > 0) {
            mergeSort(&array[displs[tid]], counts[tid]);
        }
        double t1 = omp_get_wtime();
        localSortTimes[tid] = t1 - t0;
    }

    /* Step 3: Binary-tree merge until thread 0 holds the complete sorted array */
    double merge_start = omp_get_wtime();
    int step = 1;

    while (step < numThreads) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            /* Only active threads at this binary tree level participate */
            if (tid % (2 * step) == 0) {
                int partner = tid + step;
                if (partner < numThreads && curSizes[partner] > 0) {
                    int leftLen = curSizes[tid];
                    int rightLen = curSizes[partner];
                    int totalLen = leftLen + rightLen;

                    /* Merge contiguous sub-ranges into the scratch buffer */
                    mergeTwoParts(&array[displs[tid]], leftLen,
                                  &array[displs[partner]], rightLen,
                                  &tempMerge[displs[tid]]);

                    /* Copy merged result back to array */
                    memcpy(&array[displs[tid]], &tempMerge[displs[tid]],
                           (size_t)totalLen * sizeof(int));

                    curSizes[tid] = totalLen;
                }
            }
        }
        step *= 2;
    }

    double merge_end = omp_get_wtime();
    double end_time = omp_get_wtime();

    /* Find min and max local sort times across all threads */
    double min_sort_time = localSortTimes[0];
    double max_sort_time = localSortTimes[0];
    for (int t = 1; t < numThreads; t++) {
        if (localSortTimes[t] < min_sort_time) min_sort_time = localSortTimes[t];
        if (localSortTimes[t] > max_sort_time) max_sort_time = localSortTimes[t];
    }

    /* Print sample output after sorting */
    printf("After  (first %d): ", SHOW < arraySize ? SHOW : arraySize);
    for (int i = 0; i < SHOW && i < arraySize; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    /* Timing calculations */
    double partition_time = after_partition - start_time;
    double merge_time     = merge_end - merge_start;
    double total_time     = end_time - start_time;

    /* Run sequential sort on reference array for timing & verification */
    double seq_start = omp_get_wtime();
    mergeSort(reference, arraySize);
    double seq_end = omp_get_wtime();
    double seq_time = seq_end - seq_start;

    /* Verify correctness */
    int pass = isSorted(array, arraySize);
    for (int i = 0; i < arraySize && pass; i++) {
        if (array[i] != reference[i]) {
            pass = 0;
        }
    }

    double speedup = (total_time > 0.0) ? seq_time / total_time : 0.0;
    double efficiency = (numThreads > 0) ? (speedup / numThreads) * 100.0 : 0.0;

    /* Memory estimates */
    int minChunk = counts[numThreads - 1];
    int maxChunk = counts[0];
    double localSortMemMiB = (2.0 * maxChunk * sizeof(int)) / (1024.0 * 1024.0);
    double totalMemMiB     = (3.0 * arraySize * sizeof(int)) / (1024.0 * 1024.0);

    /* Formatted results identical in style to mpi_mergesort.c */
    printf("\n=============== OPENMP MERGE SORT RESULTS ===============\n");
    printf("Threads (T)         : %d\n", numThreads);
    printf("Array Size (N)      : %d\n", arraySize);
    printf("Elements per Thread : min = %d, max = %d\n", minChunk, maxChunk);
    printf("Partition Time      : %.6f s\n", partition_time);
    printf("Local Sort (min/max): %.6f s / %.6f s\n", min_sort_time, max_sort_time);
    printf("Tree Merge Time     : %.6f s\n", merge_time);
    printf("Total Parallel Time : %.6f s\n", total_time);
    printf("Sequential Time     : %.6f s\n", seq_time);
    printf("Speedup             : %.3fx\n", speedup);
    printf("Parallel Efficiency : %.2f%%\n", efficiency);
    printf("Local Sort Mem (Est.): %.2f MiB per thread\n", localSortMemMiB);
    printf("Total Shared Peak   : %.2f MiB (approx.)\n", totalMemMiB);
    printf("Verification        : %s (N = %d, threads = %d)\n",
           pass ? "PASS" : "FAIL", arraySize, numThreads);
    printf("=========================================================\n");

    int exitStatus = pass ? 0 : 1;

    /* Clean up allocated memory */
    free(array);
    free(tempMerge);
    free(reference);
    free(counts);
    free(displs);
    free(curSizes);
    free(localSortTimes);

    return exitStatus;
}

/* Single-allocation Merge Sort wrapper (avoids repeated malloc/free in recursion) */
void mergeSort(int array[], int length) {
    if (length <= 1) {
        return;
    }
    int *temp = (int *)malloc((size_t)length * sizeof(int));
    if (temp == NULL) {
        fprintf(stderr, "Out of memory in mergeSort\n");
        exit(1);
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

/* Merge two separate sorted arrays into outArray (used in binary tree merge) */
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
