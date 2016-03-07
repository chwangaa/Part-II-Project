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
int test1(){
    int M, N, K, incRowA, incRowB, incRowC;
    M = N = 8;
    K = 384;
    incRowA = K;
    incRowB = incRowC = N;


    Dtype* A = create_matrix(8, 384, 384, 1);
    Dtype* B = create_matrix(384, 8, 8, 1);
    Dtype* C = create_matrix(8, 8, 8, 0);

    packed_mm(
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

    if(error == 0){
        fprintf(stderr, "test1 PASSES\n");
    }
    else{
        fprintf(stderr, "test1 FAILS\n");
    }
    return error;
}

int test2(){
    int M, N, K, incRowA, incRowB, incRowC;
    M = N = 8;
    K = 384;
    incRowA = K;
    incRowB = incRowC = N;


    Dtype* A = create_matrix(8, 384, 384, 1);
    Dtype* B = create_matrix(384, 8, 8, 1);
    Dtype* C = create_matrix(8, 8, 8, 0);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A[i*incRowA+j] = i;
        }
    }

    Dtype* packed_A = create_matrix(8, 384, 384, 1);
    MakePackedA(A, M, K, incRowA, M, K, packed_A);


    packed_mm(
            M, N, K,
            packed_A, incRowA,
            B, incRowB,
            C, incRowC);

    int error = 0;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != K*i){
                error++;
                fprintf(stderr, "%d %d %f \n", i, j, C[i*incRowC+j]);
            }
            }
    }

    if(error == 0){
        fprintf(stderr, "test2 PASSES\n");
    }
    else{
        fprintf(stderr, "test2 FAILS\n");
    }
    return error;
}


int test3(){
    int M, N, K, incRowA, incRowB, incRowC;
    M = N = 16;
    K = 16;
    incRowA = K;
    incRowB = incRowC = N;


    Dtype* A = create_matrix(M, K, K, 1);
    Dtype* B = create_matrix(K, N, N, 1);
    Dtype* C = create_matrix(M, N, N, 0);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A[i*incRowA+j] = i;
        }
    }

    Dtype* packed_A = create_matrix(M, K, incRowA, 1);
    

    MakePackedA(A, M, K, incRowA, M, K, packed_A);


    packed_mm(
            M, N, K,
            packed_A, incRowA,
            B, incRowB,
            C, incRowC);

    int error = 0;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != K*i){
                error++;
                fprintf(stderr, " %.0f", i, j, C[i*incRowC+j]);
            }
            }
    }

    if(error == 0){
        fprintf(stderr, "test3 PASSES\n");
    }
    else{
        fprintf(stderr, "test3 FAILS\n");
    }
    return error;
}


int test4(){
    int M, N, K, incRowA, incRowB, incRowC;
    M = N = 16;
    K = 16;
    incRowA = K;
    incRowB = incRowC = N;


    Dtype* A = create_matrix(M, K, K, 1);
    Dtype* B = create_matrix(K, N, N, 1);
    Dtype* C = create_matrix(M, N, N, 0);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A[i*incRowA+j] = i;
        }
    }

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            B[i*incRowA+j] = j;
        }
    }

    Dtype* packed_A = create_matrix(M, K, incRowA, 1);
    Dtype* packed_B = create_matrix(K, N, incRowB, 1);

    MakePackedA(A, M, K, incRowA, M, K, packed_A);
    MakePackedB(B, K, N, incRowB, K, N, packed_B);

    packed_mm(
            M, N, K,
            packed_A, incRowA,
            packed_B, incRowB,
            C, incRowC);

    int error = 0;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != K*i*j){
                error++;
                fprintf(stderr, " %.0f", i, j, C[i*incRowC+j]);
            }
            }
    }

    if(error == 0){
        fprintf(stderr, "test4 PASSES\n");
    }
    else{
        fprintf(stderr, "test4   FAILS\n");
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
    
    test1();
    test2();
    test3();    
    test4();    
    // execute_test(&test1, 100, 100, 100, "100x100x100_all_1");
    // execute_test(&test1, 513, 513, 513, "513x513x513_all_1");
    // execute_test(&test2, 513, 513, 513, "513x513x513_B_j");
    // execute_test(&test2, 1024, 1024, 1024, "501x121x123_B_j");
    // execute_test(&test1, 1213, 797, 2111, "1213x797x2111_all_1");

}