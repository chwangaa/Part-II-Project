#include "matrix.h"

void SimpleMatrixMultiplication(
    const unsigned int M,
    const unsigned int N,
    const unsigned int K,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j ++){
            Dtype accum = 0;
            for(int k = 0; k < K; k++){
                accum += A[i*incRowA+k] * B[k*incRowB+j];
            }
            C[i*incRowC+j] = accum;
        }
    }
}