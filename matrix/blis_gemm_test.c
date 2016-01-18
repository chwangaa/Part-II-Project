#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
// #include "setting.h"
#include "blis.h"
#include "util.h"
const unsigned int M_default = 16;
const unsigned int N_default = 16;
const unsigned int K_default = 16;
const unsigned int spacingFactor = 1;

typedef float Dtype;

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




    uint64_t start_time = timestamp_us();
    float alpha = 1;
    float beta = 0;
    float* alpha_ptr = &alpha;
    float* beta_ptr = &beta;

    err_t BLIS_INITIALIZED = bli_init();
    assert(BLIS_INITIALIZED == BLIS_SUCCESS);    
    auxinfo_t* auxinfo_data = (auxinfo_t*)malloc(sizeof(auxinfo_t));
    assert(auxinfo_data);
    bli_sgemm(
            BLIS_NO_TRANSPOSE,
            BLIS_NO_TRANSPOSE,
            M, N, K,
            alpha_ptr,
            A, incRowA, 1,
            B, incRowB, 1,
            beta_ptr,
            C, incRowC, 1);


    err_t bli_finalize(void);
    uint64_t end_time = timestamp_us();
    double m_second_taken = (double)(end_time - start_time) / 1000.0;
    int error = 0;
    for(int i = 0; i < 8; i++){
    // 	// fprintf(stderr, "%d \n", fix16_to_int(M3[i]));
    	for(int j = 0; j < 8; j++){
            if(C[i*incRowC+j] != K){
                // error++;
                fprintf(stderr, "%d %d %f \n", i, j, C[i*incRowC+j]);
            }
    		    // fprintf(stderr, "%d ", (int)C[i*incRowC+j]);
            }
        // fprintf(stderr, "\n");
    }
    printf("%d, %d, %d, %d, %f \n", M, N, K, error, m_second_taken);
}