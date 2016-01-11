#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include "StrassenMatrixMultiplication.h"
#include "util.h"
#include "setting.h"
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

	Dtype* A = (Dtype*)malloc(sizeof(Dtype)*M*incRowA);
	Dtype* B = (Dtype*)malloc(sizeof(Dtype)*K*incRowB);
	Dtype* C = (Dtype*)malloc(sizeof(Dtype)*M*incRowC);

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


    // float temp[9] = {2, 2, 1, 2, 2, 1, 1, 1, 1};
    // float temp1[2] = {2,1};
    // float temp2[4] = {2, 2, 2, 2};
    // A = temp1;
    // B = temp2;

    uint64_t start_time = timestamp_us();
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
            if(C[i*incRowC+j] != K){
                error++;
                // fprintf(stderr, "%d %d %f \n", i, j, C[i*incRowC+j]);
            }
    		    // fprintf(stderr, "%d ", (int)C[i*incRowC+j]);
            }
        // fprintf(stderr, "\n");
    }
    printf("%d, %d, %d, %d, %f \n", M, N, K, error, m_second_taken);
}