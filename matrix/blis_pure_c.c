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
    Dtype* A = (Dtype*)malloc_aligned2(32, sizeof(Dtype)*M*K);
    Dtype* B = (Dtype*)malloc_aligned2(32, sizeof(int)*K*incRowB);
    Dtype* C = (Dtype*)malloc_aligned2(32, sizeof(int)*M*incRowC);
    assert(A);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            // fprintf(stderr, "%d, %d \n", i, j);
            A[i*incRowA+j] = 1;
        }
    }
    
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            B[i*incRowB+j] = 1;
        }
    }

    
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            C[i*incRowC+j] = 0;
        }
    }

    
    // bli_sgemm(
    //         BLIS_NO_TRANSPOSE,
    //         BLIS_NO_TRANSPOSE,
    //         M, N, K,
    //         alpha_ptr,
    //         A, incRowA, 1,
    //         B, incRowB, 1,
    //         beta_ptr,
    //         C, incRowC, 1);
    float alpha = 1;
    float beta = 0;
    float* alpha_ptr = &alpha;
    float* beta_ptr = &beta;
    // auxinfo_t* auxinfo_data = (auxinfo_t*)malloc(sizeof(auxinfo_t));
    // assert(auxinfo_data);
    // assert((unsigned long)_A % 16 == 0);
    // bli_sgemm_asm_8x8(K, alpha_ptr,
    //             A,
    //             B,
    //             beta_ptr,
    //             C,
    //             incRowC, 
    //             1,
    //             auxinfo_data);    


    uint64_t start_time = timestamp_us();


    // dgemm_nn(
    //         M, N, K,
    //         A, incRowA,
    //         B, incRowB,
    //         C, incRowC);
    uint64_t end_time = timestamp_us();
    double m_second_taken = (double)(end_time - start_time) / 1000.0;
    int error = 0;
    for(int i = 0; i < M; i++){
    //  // fprintf(stderr, "%d \n", fix16_to_int(M3[i]));
        for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != K){
                error++;
                // fprintf(stderr, "%d %d %f \n", i, j, C[i*incRowC+j]);
            }
                // fprintf(stderr, "%d ", (int)C[i*incRowC+j]);
            }
        // fprintf(stderr, "\n");
    }
    printf("%d, %d, %d, %d, %f \n", M, N, K, error, m_second_taken);
    free_aligned2(A);
    free_aligned2(B);
    free_aligned2(C);  
}

#endif