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

    switch(ALGORITHM){
        case 0:
        blis_mm(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
        break;
        case 1:
        cblas_mm(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
        break;
        case 2:
        packed_mm(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
        break;
        case 3:
        packed_strassen_mm(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
        break;
        default:
        strassen_mm(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
    }
}

double benchmark_matrix_addition(int M, int N, int repeat){
    Dtype* A = create_matrix(M, N, N, 1);
    Dtype* B = create_matrix(M, N, N, 1);
    Dtype* C = create_matrix(M, N, N, 1);

    uint64_t start_time = timestamp_us();

    for(int i = 0; i < repeat; i++){
        matrix_addition(M, N,
                        A, N,
                        B, N,
                        C, N);
    }
    uint64_t end_time = timestamp_us();
    double dt = (double)(end_time-start_time) / (1000.0*repeat);   
    return dt;
}

void benchmark_mm(int M, int N, int K, int n, int ALGO){
    uint64_t start_time = timestamp_us();
    uint64_t clk_start = clock();
    for(int i = 0; i < n; i++){
        single_test(M, N, K, ALGO);
    }
    double clk = (double)(clock() - clk) / n;
    uint64_t end_time = timestamp_us();
    double dt = (double)(end_time-start_time) / (1000.0*n);
    switch(ALGO){
        case 0:
        fprintf(stderr, "BLIS takes %lf ms and %lf to complete %dx%dx%d\n", dt, clk, M, N, K);
        break;
        case 1:
        fprintf(stderr, "CBLAS takes %lf ms and %lf to complete %dx%dx%d\n", dt, clk, M, N, K);
        break;
        case 2:
        fprintf(stderr, "PACKED takes %lf ms and %lf to complete %dx%dx%d\n", dt, clk, M, N, K);
        break;
        case 3:
        fprintf(stderr, "PACKED_STRASSEN takes %lf ms and %lf to complete %dx%dx%d\n", dt, clk, M, N, K);
        break;
        default:
        fprintf(stderr, "STRASSEN takes %lf ms and %lf to complete %dx%dx%d\n", dt, clk, M, N, K);
    }
}