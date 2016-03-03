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
int test1(int M, int N, int K){
    int incRowA, incRowB, incRowC;
    incRowA = K;
    incRowB = incRowC = N;

    Dtype* A = create_matrix(M, K, incRowA, 1);
    Dtype* B = create_matrix(K, N, incRowB, 1);
    Dtype* C = create_matrix(M, N, incRowC, 0);

    matrix_multiplication(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);

    int error = 0;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != K){
                error++;
                fprintf(stderr, "%d %d %f \n", i, j, C[i*incRowC+j]);
            }
            }
    }

    return error;
}

/*
    this test case makes a matrix multiplication of size (M, N, K)
    matrix A has all entries being 1
    matrix B has the entries: B[i, j] = j
    the expected value of C is: C[i, j] = j * K
*/
int test2(int M, int N, int K){
    int incRowA, incRowB, incRowC;
    incRowA = K;
    incRowB = incRowC = N;

    Dtype* A = create_matrix(M, K, incRowA, 1);
    Dtype* B = create_matrix(K, N, incRowB, 0);
    Dtype* C = create_matrix(M, N, incRowC, 0);

    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            B[i*incRowB+j] = j;
        }
    }

    matrix_multiplication(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);

    int error = 0;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != K*j){
                error++;
                fprintf(stderr, "%d %d %f \n", i, j, C[i*incRowC+j]);
            }
            }
    }

    return error;
}

void execute_test(void* test, int M, int N, int K, char* test_name){
    int (*test_func)(int, int, int) = test;
    if(!(*test_func)(M, N, K)){
        fprintf(stderr, "%s PASSES\n", test_name);
    }
    else{
        fprintf(stderr, "%d FAILS\n", test_name);
    }
}

int main(int argc, char** argv) {
    
    execute_test(&test1, 100, 100, 100, "100x100x100_all_1");
    execute_test(&test1, 513, 513, 513, "513x513x513_all_1");
    execute_test(&test2, 513, 513, 513, "513x513x513_B_j");
    execute_test(&test2, 1024, 1024, 1024, "501x121x123_B_j");
    execute_test(&test1, 1213, 797, 2111, "1213x797x2111_all_1");

}