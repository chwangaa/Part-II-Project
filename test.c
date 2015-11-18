#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
// #include "SimpleMatrixMultiplication.h"
#include "matrix/matrix.h"
// #include "CacheObliviousMatrixMultiplication.h"
// #include "StrassenMatrixMultiplication.h"
#include "util.h"
const unsigned int M_default = 16;
const unsigned int N_default = 16;
const unsigned int K_default = 16;

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
    int incRowA = K;
    int incRowB = N;
    int incRowC = N;

	Dtype* A = (Dtype*)malloc(sizeof(int)*M*incRowA);
	Dtype* B = (Dtype*)malloc(sizeof(int)*K*incRowB);
	Dtype* C = (Dtype*)malloc(sizeof(int)*M*incRowC);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
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




    uint64_t start_time = timestamp_us();
    // SimpleMatrixMultiplication(
    //         M, N, K,
    //         A, incRowA,
    //         B, incRowB,
    //         C, incRowC);
    // cblas_gemm(
    //         M, N, K,
    //         A, incRowA,
    //         B, incRowB,
    //         C, incRowC);
    // cache_oblivious_matrix_multiplication(
    //         M, N, K,
    //         A, incRowA,
    //         B, incRowB,
    //         C, incRowC);
    strassen_matrix_multiplication(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
    uint64_t end_time = timestamp_us();
    double m_second_taken = (double)(end_time - start_time) / 1000.0;
    int error = 0;
    for(int i = 0; i < M; i++){
    // 	// fprintf(stderr, "%d \n", fix16_to_int(M3[i]));
    	for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != (float)M){
                error++;
            }
    		    // fprintf(stderr, "%d ", C[i*incRowC+j]);
            }
        // fprintf(stderr, "\n");
    }
    // printf("M,  N,  K,  error, Time taken \n");
    printf("%d, %d, %d, %d, %f \n", M, N, K, error, m_second_taken);
}