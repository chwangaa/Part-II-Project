#ifndef CBLAS_GEMM_H
#define CBLAS_GEMM_H

#include "setting.h"
#include <cblas.h>

void cblas_gemm(
          const unsigned int M, 
          const unsigned int N, 
          const unsigned int K,
     			const Dtype *A, const int incRowA,
     			const Dtype *B, const int incRowB,
     			Dtype *C, const int incRowC){
	return cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		M, N, K,
		1, 
		A, K, 
		B, N, 
		1, 
		C, N);
}

#endif