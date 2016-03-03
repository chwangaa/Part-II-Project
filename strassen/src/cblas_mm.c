#include <cblas.h>
#include "matrix.h"


void cblas_mm(
          const unsigned int M, 
          const unsigned int N, 
          const unsigned int K,
     			const Dtype *A, const int incRowA,
     			const Dtype *B, const int incRowB,
     			Dtype *C, const int incRowC){
	return cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		M, N, K,
		1, 
		A, incRowA, 
		B, incRowB, 
		0, 
		C, incRowC);
}