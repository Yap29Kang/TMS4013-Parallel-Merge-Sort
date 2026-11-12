#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

void mergeSort(int array[], int length);
void merge(int leftArray[], int leftSize, int rightArray[], int rightSize, int array[]);

// High-resolution clock: returns microseconds as a double
double now_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double) counter.QuadPart * 1000000.0 / (double) freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
#endif
}

int main () {
    int array[10];
    int arraySize = sizeof(array) / sizeof(array[0]);

    // Fill the array with random numbers between 0 and arraySize - 1
    for (int i = 0; i < arraySize; i++) {
        array[i] = rand() % arraySize;
    }

    // Before
    for (int i = 0; i < arraySize; i++) {
   		printf("%d ", array[i]);	
	}

    printf("\n===================================================\n");

    // Start timer
    double start_time = now_us();

    // Sort
    mergeSort(array, arraySize);

    // End timer
    double end_time = now_us();

    // After
   	for (int i = 0; i < arraySize; i++) {
   		printf("%d ", array[i]);	
	}
	printf("\n");

    double seconds = (end_time - start_time) / 1000000.0;
    printf("\nProgram execution time: %.6f s\n", seconds);

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
