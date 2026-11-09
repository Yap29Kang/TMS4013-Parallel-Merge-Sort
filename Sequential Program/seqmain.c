#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

void quicksort(int array[], int start, int end);
int partition(int array[], int start, int end);

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
    int array[10000];
    int arraySize = sizeof(array) / sizeof(array[0]);

    // Fill the array with random numbers between 0 and 9999
    for (int i = 0; i < arraySize; i++) {
        array[i] = rand() % 10000;
    }

    // Before
    for (int i = 0; i < arraySize; i++) {
        printf("This is item %d which is %d\n", i, array[i]);
    }

    printf("===================================================\n");

	// Start timer
    double start_time = now_us();                

	// Sort
    quicksort(array, 0, arraySize - 1);

	// End timer
    double end_time = now_us();

    // After
    for (int i = 0; i < arraySize; i++) {
        printf("This is item %d which is %d\n", i, array[i]);
    }

    double seconds = (end_time - start_time) / 1000000.0;
    printf("Program execution time: %.6f s\n", seconds);

    return 0;
}

void quicksort (int array[], int start, int end) {
    if (end <= start) return; // base case

    int pivot = partition(array, start, end);
    quicksort(array, start, pivot - 1);
    quicksort(array, pivot + 1, end);
}

int partition (int array[], int start, int end) {
    int pivot = array[end];
    int i = start - 1;

    for (int j = start; j <= end - 1; j++) {
        if (array[j] < pivot) {
            i++;
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    i++;
    int temp = array[i];
    array[i] = array[end];
    array[end] = temp;

    return i;
}
