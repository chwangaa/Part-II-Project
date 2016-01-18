/*! \file gemm.h
    \brief none-assembly version of GEMM algorithm for FIXED-POINT datatype
*/
#ifndef GEMM_H
#define GEMM_H
// #include "setting.h"
#include <stdio.h>

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include "util.h"
#include "src/matrix_util.h"

const unsigned int M_default = 16;
const unsigned int N_default = 16;
const unsigned int K_default = 16;
const unsigned int spacingFactor = 1;

int main(int argc, char** argv) {
  int M, N, K;
  if (argc < 4) {
    fprintf(stderr, "M, N, K not given, use the default values\n");
    M = M_default;
    N = N_default;
    K = K_default;
  }
  else{
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }
    int incRowA = K * spacingFactor;
    int incRowB = N * spacingFactor;
    int incRowC = N * spacingFactor;

    // Dtype* restrict A = (Dtype*)malloc(sizeof(Dtype)*M*incRowA);
    // Dtype* restrict B = (Dtype*)malloc(sizeof(Dtype)*K*incRowB);
    // Dtype* restrict C = (Dtype*)malloc(sizeof(Dtype)*M*incRowC);
    // Dtype A[M*incRowA];
    // Dtype B[K*incRowB];
    // Dtype C[M*incRowC];
    Dtype* A = (Dtype*)malloc_aligned(32, sizeof(Dtype)*M*K);
    assert(A);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            // fprintf(stderr, "%d, %d \n", i, j);
            A[i*incRowA+j] = j;
        }
    }

    Dtype* newA = (Dtype*)malloc_aligned(32, sizeof(Dtype)*M*K);
    MakePackedA(A, M, K, incRowA, 
                16, 16,
                newA);   
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            fprintf(stderr, "%d ", (int)newA[i*incRowA+j]);
        }
        fprintf(stderr, "\n");
    }
}

#endif