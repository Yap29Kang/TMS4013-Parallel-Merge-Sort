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
 *   - Thread 0 (T0) ends up holding the complete sorted array.
 *
 * Input (Command-Line Arguments):
 *   ./openmp_mergesort [N] [T]
 *     N : Optional number of integers to sort (default: 1000000)
 *     T : Optional number of OpenMP threads (default: omp_get_max_threads())
 *
 * Output:
 *   - Sample elements before & after sorting (first 10 elements)
 *   - Thread trace (only when N <= 32): boxed table showing each thread's
 *     chunk, the local sort stages, and which thread merges with which
 *     at each tree-merge level (e.g. T0 <- T1)
 *   - Thread work breakdown: chunk range, element count, OpenMP thread id
 *     and local sort time for each chunk, plus the merge schedule per level
 *   - Load balance summary (min/max elements per thread)
 *   - Timing breakdown (Partition, Local Sort min/max, Tree Merge, Total, Sequential)
 *   - Speedup, Parallel Efficiency, and estimated Memory Usage (MiB)
 *   - Verification result (PASS/FAIL against ascending check & sequential sort)
 *
 * Compilation:
 *   gcc -O2 -Wall -fopenmp -o openmp_mergesort openmp_mergesort.c
 *
 * Execution:
 *   ./openmp_mergesort 16 4          (small visual test with thread trace)
 *   ./openmp_mergesort 10 3          (odd thread count, shows the "carry" case)
 *   ./openmp_mergesort 10000000 4    (scaling benchmark)
 * ============================================================================
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <omp.h>
 
 #define SHOW 10        /* number of elements printed in the Before/After samples */
 #define TRACE_MAX 32   /* print the per-thread trace only when N <= this */
 #define BUFSZ 4096     /* text buffer size used by the trace printer */
 
 void mergeSort(int array[], int length);
 void mergeSortRec(int array[], int temp[], int left, int right);
 void merge(int array[], int temp[], int left, int middle, int right);
 void mergeTwoParts(int leftArray[], int leftSize, int rightArray[], int rightSize, int outArray[]);
 int isSorted(int array[], int length);
 
 /* ========================================================================
  * TRACE HELPERS
  * These functions only draw the boxed "which thread does what" table for
  * small arrays. They do not take part in the sorting itself.
  * ======================================================================== */
 
 /* Comparison function for qsort (used only to draw the illustrative stages) */
 static int cmpInt(const void *a, const void *b) {
     int x = *(const int *)a, y = *(const int *)b;
     return (x > y) - (x < y);
 }
 
 /* Format n ints grouped in runs of runw (runs separated by " / ").
  * If sortRuns is set, each run is sorted first (shows the bottom-up merge stages). */
 static void fmtRuns(char *buf, const int *src, int n, int runw, int sortRuns) {
     size_t len = 0;
     int *tmp = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
     if (runw < 1) runw = 1;
     if (n > 0) memcpy(tmp, src, (size_t)n * sizeof(int));
     buf[0] = '\0';
     for (int s = 0; s < n; s += runw) {
         int e = (s + runw < n) ? s + runw : n;
         if (sortRuns) qsort(tmp + s, (size_t)(e - s), sizeof(int), cmpInt);
         if (s > 0) len += (size_t)snprintf(buf + len, BUFSZ - len, " / ");
         for (int i = s; i < e; i++)
             len += (size_t)snprintf(buf + len, BUFSZ - len, "%s%d", i > s ? " " : "", tmp[i]);
     }
     free(tmp);
 }
 
 /* Print a horizontal border line. Cells at the current level span 'span' columns. */
 static void sepLine(int p, int cellW, int T, int span) {
     if (!p) return;                       /* p = 0 is the measuring pass: print nothing */
     putchar('+');
     for (int c = 0; c < T; c += span) {
         int k = (span < T - c) ? span : T - c;
         int n = k * cellW + 3 * (k - 1) + 2;
         for (int i = 0; i < n; i++) putchar('-');
         putchar('+');
     }
     putchar('\n');
 }
 
 /* Print one table cell spanning k columns, centred in its width.
  * Also records the column width needed to fit the text (used by the measuring pass). */
 static void putCell(int p, int *need, int k, int cellW, const char *s) {
     int len = (int)strlen(s);
     int req = (len - 3 * (k - 1) + k - 1) / k;
     if (req > *need) *need = req;
     if (p) {
         int width = k * cellW + 3 * (k - 1);
         int pad = width - len;
         if (pad < 0) pad = 0;
         int l = pad / 2;
         printf(" %*s%s%*s |", l, "", s, pad - l, "");
     }
 }
 
 /* Draws the whole table. Runs twice: p=0 measures the needed column width,
  * p=1 prints. snaps[0] = array after local sort, snaps[l] = array after merge level l. */
 static int renderAll(int p, int cellW, const int *orig, int T, const int *counts,
                      const int *displs, int *const *snaps, int levels) {
     int need = 1;
     char buf[BUFSZ], lab[64];
     int maxChunk = 0;
     for (int t = 0; t < T; t++) if (counts[t] > maxChunk) maxChunk = counts[t];
 
     /* Header row: thread names T0..T(n-1) */
     sepLine(p, cellW, T, 1);
     if (p) putchar('|');
     for (int t = 0; t < T; t++) { snprintf(lab, sizeof lab, "T%d", t); putCell(p, &need, 1, cellW, lab); }
     if (p) putchar('\n');
 
     /* Row: the raw (unsorted) chunk each thread received */
     if (p) putchar('|');
     for (int t = 0; t < T; t++) { fmtRuns(buf, orig + displs[t], counts[t], counts[t], 0); putCell(p, &need, 1, cellW, buf); }
     if (p) putchar('\n');
 
     /* Rows: illustrative bottom-up sorting stages (runs of 2, 4, ...) inside each chunk */
     for (int w = 2; w < maxChunk; w *= 2) {
         if (p) putchar('|');
         for (int t = 0; t < T; t++) { fmtRuns(buf, orig + displs[t], counts[t], w, 1); putCell(p, &need, 1, cellW, buf); }
         if (p) putchar('\n');
     }
 
     /* Row: real result after each thread's local sort */
     if (p) putchar('|');
     for (int t = 0; t < T; t++) { fmtRuns(buf, snaps[0] + displs[t], counts[t], counts[t], 0); putCell(p, &need, 1, cellW, buf); }
     if (p) putchar('\n');
 
     /* Tree-merge levels: each level halves the number of active threads */
     int prevSpan = 1;
     for (int lvl = 1; lvl <= levels; lvl++) {
         int step = 1 << (lvl - 1), span = 2 * step;
         sepLine(p, cellW, T, prevSpan);
 
         /* Label row: who merges with whom ("T0 <- T1" or "T2 (carry)") */
         if (p) putchar('|');
         for (int c = 0; c < T; c += span) {
             int k = (span < T - c) ? span : T - c;
             if (c + step < T) snprintf(lab, sizeof lab, "T%d <- T%d", c, c + step);
             else              snprintf(lab, sizeof lab, "T%d (carry)", c);
             putCell(p, &need, k, cellW, lab);
         }
         if (p) putchar('\n');
 
         /* Data row: merged result held by the receiving thread */
         if (p) putchar('|');
         for (int c = 0; c < T; c += span) {
             int k = (span < T - c) ? span : T - c;
             int end = c + k;
             int n = displs[end - 1] + counts[end - 1] - displs[c];
             fmtRuns(buf, snaps[lvl] + displs[c], n, n, 0);
             putCell(p, &need, k, cellW, buf);
         }
         if (p) putchar('\n');
         prevSpan = span;
     }
     sepLine(p, cellW, T, prevSpan);
     return need;
 }
 
 /* Print the full trace: input line, then the table (measure pass, then print pass) */
 static void printTrace(const int *orig, int N, int T, const int *counts,
                        const int *displs, int *const *snaps, int levels) {
     printf("\nInput  : ");
     for (int i = 0; i < N; i++) printf("%d ", orig[i]);
     printf("\nThreads: T0..T%d, one chunk each ('/' separates sorted runs)\n", T - 1);
     int cellW = renderAll(0, 1, orig, T, counts, displs, snaps, levels);
     renderAll(1, cellW, orig, T, counts, displs, snaps, levels);
 }
 
 /* ========================================================================
  * MAIN
  * ======================================================================== */
 
 int main(int argc, char *argv[]) {
     /* Read array size from command line (default: 1,000,000) */
     int arraySize = (argc > 1) ? atoi(argv[1]) : 1000000;
     if (arraySize <= 0) { fprintf(stderr, "Error: array size must be > 0\n"); return 1; }
 
     /* Read thread count from command line (default: max available threads) */
     int numThreads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
     if (numThreads <= 0) numThreads = 1;
     omp_set_dynamic(0);              /* do not let the runtime shrink the thread team */
     omp_set_num_threads(numThreads);
 
     /* Allocate: working array, scratch buffer for the tree merge, sequential reference */
     int *array     = (int *)malloc((size_t)arraySize * sizeof(int));
     int *tempMerge = (int *)malloc((size_t)arraySize * sizeof(int));
     int *reference = (int *)malloc((size_t)arraySize * sizeof(int));
     if (!array || !tempMerge || !reference) {
         fprintf(stderr, "Error: Out of memory for N = %d\n", arraySize);
         free(array); free(tempMerge); free(reference);
         return 1;
     }
 
     /* Fixed seed ensures identical dataset across different thread counts & MPI comparison */
     srand(42);
     for (int i = 0; i < arraySize; i++) {
         array[i] = rand() % arraySize;
         reference[i] = array[i];     /* copy for the sequential baseline */
     }
 
     printf("Before (first %d): ", SHOW < arraySize ? SHOW : arraySize);
     for (int i = 0; i < SHOW && i < arraySize; i++) printf("%d ", array[i]);
     printf("\n===================================================\n");
 
     /* Per-thread bookkeeping:
      *   counts[t]         = number of elements in thread t's chunk
      *   displs[t]         = start index of thread t's chunk in the array
      *   curSizes[t]       = current size of the sorted block thread t holds
      *   ranOn[t]          = OpenMP thread id that actually sorted chunk t
      *   localSortTimes[t] = time chunk t took to sort locally */
     int *counts   = (int *)malloc((size_t)numThreads * sizeof(int));
     int *displs   = (int *)malloc((size_t)numThreads * sizeof(int));
     int *curSizes = (int *)malloc((size_t)numThreads * sizeof(int));
     int *ranOn    = (int *)malloc((size_t)numThreads * sizeof(int));
     double *localSortTimes = (double *)malloc((size_t)numThreads * sizeof(double));
     if (!counts || !displs || !curSizes || !ranOn || !localSortTimes) {
         fprintf(stderr, "Error: Out of memory allocating thread metadata\n");
         return 1;
     }
 
     /* Load balancing: distribute elements evenly (chunk sizes differ by at most 1).
      * The first 'extra' threads receive one additional element. */
     int base = arraySize / numThreads, extra = arraySize % numThreads, offset = 0;
     for (int t = 0; t < numThreads; t++) {
         counts[t] = base + (t < extra ? 1 : 0);
         displs[t] = offset;
         curSizes[t] = counts[t];
         localSortTimes[t] = 0.0;
         ranOn[t] = -1;
         offset += counts[t];
     }
 
     /* Number of tree-merge levels = ceil(log2(T)) */
     int levels = 0;
     for (int s = 1; s < numThreads; s *= 2) levels++;
 
     /* Trace storage (small N only): original data + a snapshot after each phase */
     int trace = (arraySize <= TRACE_MAX);
     int *orig = NULL;
     int **snaps = NULL;
     if (trace) {
         orig = (int *)malloc((size_t)arraySize * sizeof(int));
         snaps = (int **)malloc((size_t)(levels + 1) * sizeof(int *));
         memcpy(orig, array, (size_t)arraySize * sizeof(int));
         for (int l = 0; l <= levels; l++) snaps[l] = (int *)malloc((size_t)arraySize * sizeof(int));
     }
 
     /* Start parallel timer */
     double start_time = omp_get_wtime();
 
     /* Step 1: Chunk mapping / partition (in shared memory this costs nothing) */
     double after_partition = omp_get_wtime();
 
     /* Step 2: Local sort. Thread t sorts chunk t on its own contiguous part of the array.
      * parallel for + schedule(static,1) gives each chunk to a different thread, and
      * num_threads(numThreads) ensures every chunk is handled even if the runtime
      * would otherwise create fewer threads. */
     #pragma omp parallel for num_threads(numThreads) schedule(static, 1)
     for (int t = 0; t < numThreads; t++) {
         double t0 = omp_get_wtime();
         ranOn[t] = omp_get_thread_num();                 /* remember which thread did this chunk */
         if (counts[t] > 0) mergeSort(&array[displs[t]], counts[t]);
         localSortTimes[t] = omp_get_wtime() - t0;
     }
     if (trace) memcpy(snaps[0], array, (size_t)arraySize * sizeof(int));
 
     /* Step 3: Binary-tree merge until T0 holds the complete sorted array.
      * Level 1: T0<-T1, T2<-T3, ...  Level 2: T0<-T2, T4<-T6, ...  and so on.
      * The implicit barrier at the end of each parallel for makes sure a level
      * is finished before the next one starts. */
     double merge_start = omp_get_wtime();
     int step = 1, lvl = 0;
     while (step < numThreads) {
         #pragma omp parallel for num_threads(numThreads) schedule(static, 1)
         for (int t = 0; t < numThreads; t++) {
             /* Only the receiving threads at this tree level do work */
             if (t % (2 * step) == 0) {
                 int partner = t + step;
                 if (partner < numThreads && curSizes[partner] > 0) {
                     int leftLen = curSizes[t], rightLen = curSizes[partner];
                     int totalLen = leftLen + rightLen;
 
                     /* Merge the two neighbouring sorted blocks into the scratch buffer */
                     mergeTwoParts(&array[displs[t]], leftLen,
                                   &array[displs[partner]], rightLen,
                                   &tempMerge[displs[t]]);
 
                     /* Copy the merged result back into the main array */
                     memcpy(&array[displs[t]], &tempMerge[displs[t]], (size_t)totalLen * sizeof(int));
                     curSizes[t] = totalLen;
                 }
             }
         }
         lvl++;
         if (trace) memcpy(snaps[lvl], array, (size_t)arraySize * sizeof(int));
         step *= 2;
     }
     double merge_end = omp_get_wtime();
     double end_time = merge_end;
 
     /* Find min and max local sort times across all threads (load balance check) */
     double min_sort_time = localSortTimes[0], max_sort_time = localSortTimes[0];
     for (int t = 1; t < numThreads; t++) {
         if (localSortTimes[t] < min_sort_time) min_sort_time = localSortTimes[t];
         if (localSortTimes[t] > max_sort_time) max_sort_time = localSortTimes[t];
     }
 
     /* Print sample output after sorting */
     printf("After  (first %d): ", SHOW < arraySize ? SHOW : arraySize);
     for (int i = 0; i < SHOW && i < arraySize; i++) printf("%d ", array[i]);
     printf("\n");
 
     /* Timing calculations */
     double partition_time = after_partition - start_time;
     double merge_time = merge_end - merge_start;
     double total_time = end_time - start_time;
 
     /* Run the sequential sort on the reference copy for timing & verification */
     double seq_start = omp_get_wtime();
     mergeSort(reference, arraySize);
     double seq_time = omp_get_wtime() - seq_start;
 
     /* Verify correctness: result must be ascending AND identical to the sequential result */
     int pass = isSorted(array, arraySize);
     for (int i = 0; i < arraySize && pass; i++)
         if (array[i] != reference[i]) pass = 0;
 
     /* Speedup = sequential time / parallel time; efficiency = speedup / threads */
     double speedup = (total_time > 0.0) ? seq_time / total_time : 0.0;
     double efficiency = (speedup / numThreads) * 100.0;
 
     /* Memory estimates: each thread needs its chunk + a same-size temp buffer;
      * overall the program holds three N-sized arrays (array, tempMerge, reference) */
     int minChunk = counts[numThreads - 1], maxChunk = counts[0];
     double localSortMemMiB = (2.0 * maxChunk * sizeof(int)) / (1024.0 * 1024.0);
     double totalMemMiB = (3.0 * arraySize * sizeof(int)) / (1024.0 * 1024.0);
 
     /* ---- Which thread did which part ---- */
     if (trace) printTrace(orig, arraySize, numThreads, counts, displs, snaps, levels);
 
     if (numThreads <= 64) {
         printf("\n---------------- THREAD WORK BREAKDOWN ----------------\n");
         printf("Phase 1 - local sort (each thread sorts its own chunk):\n");
         for (int t = 0; t < numThreads; t++)
             printf("  Chunk %-3d [%d..%d) %d elems | ran on OMP thread %d | %.6f s\n",
                    t, displs[t], displs[t] + counts[t], counts[t], ranOn[t], localSortTimes[t]);
         printf("Phase 2 - tree merge:\n");
         int st = 1, lv = 1;
         while (st < numThreads) {
             printf("  Level %d:", lv);
             for (int t = 0; t < numThreads; t += 2 * st)
                 if (t + st < numThreads) printf("  T%d <- T%d", t, t + st);
             printf("\n");
             st *= 2; lv++;
         }
         printf("  Final sorted array held by T0\n");
     }
 
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
     printf("Verification        : %s (N = %d, threads = %d)\n", pass ? "PASS" : "FAIL", arraySize, numThreads);
     printf("=========================================================\n");
 
     /* Clean up allocated memory */
     if (trace) {
         for (int l = 0; l <= levels; l++) free(snaps[l]);
         free(snaps); free(orig);
     }
     free(array); free(tempMerge); free(reference);
     free(counts); free(displs); free(curSizes); free(ranOn); free(localSortTimes);
     return pass ? 0 : 1;
 }
 
 /* ========================================================================
  * MERGE SORT ROUTINES
  * ======================================================================== */
 
 /* Single-allocation merge sort wrapper (avoids repeated malloc/free in recursion).
  * Each thread calls this on its own chunk, so the temp buffers never overlap. */
 void mergeSort(int array[], int length) {
     if (length <= 1) return;
     int *temp = (int *)malloc((size_t)length * sizeof(int));
     if (temp == NULL) { fprintf(stderr, "Out of memory in mergeSort\n"); exit(1); }
     mergeSortRec(array, temp, 0, length - 1);
     free(temp);
 }
 
 /* Recursive divide step: split in half, sort each half, then merge them */
 void mergeSortRec(int array[], int temp[], int left, int right) {
     if (left >= right) return;
     int middle = left + (right - left) / 2;
     mergeSortRec(array, temp, left, middle);
     mergeSortRec(array, temp, middle + 1, right);
     merge(array, temp, left, middle, right);
 }
 
 /* Merge the two sorted halves [left..middle] and [middle+1..right] in place via temp */
 void merge(int array[], int temp[], int left, int middle, int right) {
     int i = left, l = left, r = middle + 1;
     while (l <= middle && r <= right) {          /* take the smaller front element */
         if (array[l] <= array[r]) temp[i++] = array[l++];
         else                      temp[i++] = array[r++];
     }
     while (l <= middle) temp[i++] = array[l++];  /* copy leftovers from left half */
     while (r <= right)  temp[i++] = array[r++];  /* copy leftovers from right half */
     for (int k = left; k <= right; k++) array[k] = temp[k];
 }
 
 /* Merge two separate sorted arrays into outArray (used in the binary-tree merge) */
 void mergeTwoParts(int leftArray[], int leftSize, int rightArray[], int rightSize, int outArray[]) {
     int i = 0, l = 0, r = 0;
     while (l < leftSize && r < rightSize) {
         if (leftArray[l] <= rightArray[r]) outArray[i++] = leftArray[l++];
         else                               outArray[i++] = rightArray[r++];
     }
     while (l < leftSize)  outArray[i++] = leftArray[l++];
     while (r < rightSize) outArray[i++] = rightArray[r++];
 }
 
 /* Check that the array is in ascending order (returns 1 if sorted, 0 otherwise) */
 int isSorted(int array[], int length) {
     for (int i = 0; i < length - 1; i++)
         if (array[i] > array[i + 1]) return 0;
     return 1;
 }
 
