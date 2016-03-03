#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include "matrix.h"



/*
    this test case makes a matrix multiplication of size (M, N, K) where all entries are 1
*/
void single_test(int M, int N, int K, int ALGORITHM){
    int incRowA, incRowB, incRowC;
    incRowA = K;
    incRowB = incRowC = N;

    Dtype* A = create_matrix(M, K, incRowA, 1);
    Dtype* B = create_matrix(K, N, incRowB, 1);
    Dtype* C = create_matrix(M, N, incRowC, 0);

    if(ALGORITHM == 0){
        blis_mm(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
    }
    else{
        strassen_mm(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
    }
}

void benchmark_mm(int M, int N, int K, int n, int ALGO){
    uint64_t start_time = timestamp_us();
    for(int i = 0; i < n; i++){
        single_test(M, N, K, ALGO);
    }
    uint64_t end_time = timestamp_us();
    double dt = (double)(end_time-start_time) / (1000.0*n);
    if(ALGO == 0){
        fprintf(stderr, "BLIS takes %lf ms to complete %dx%dx%d\n", dt, M, N, K);
    }
    else{
        fprintf(stderr, "STRASSEN takes %lf ms to complete %dx%dx%d\n", dt, M, N, K);
    }
}