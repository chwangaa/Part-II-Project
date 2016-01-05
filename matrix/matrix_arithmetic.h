#ifndef MATRIX_ARITHMETIC_H
#define MATRIX_ARITHMETIC_H
#include "setting.h"

void matrix_addition(
    const unsigned int M,
    const unsigned int N,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

    Dtype* A_base;
    Dtype* B_base;
    Dtype* C_base;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            C[i*incRowC+j] = A[i*incRowA+j] + B[i*incRowB+j];
        }
    }
}

void matrix_subtraction(
    const unsigned int M,
    const unsigned int N,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j ++){
            C[i*incRowC+j] = A[i*incRowA+j] - B[i*incRowB+j];
        }
    }
}
#endif