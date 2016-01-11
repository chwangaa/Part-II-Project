#ifndef CBLAS_GEMM_H
#define CBLAS_GEMM_H

#include "../setting.h"
#include "blis.h"

// void cblas_gemm(
//           const unsigned int M, 
//           const unsigned int N, 
//           const unsigned int K,
//      			const Dtype *A, const int incRowA,
//      			const Dtype *B, const int incRowB,
//      			Dtype *C, const int incRowC){
// 	return bli_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
// 		M, N, K,
// 		1, 
// 		A, incRowA, 
// 		B, incRowB, 
// 		0, 
// 		C, incRowC);
// }

void cblas_gemm(
        const unsigned int M, 
        const unsigned int N, 
        const unsigned int K,
     	const Dtype *A, const int incRowA,
     	const Dtype *B, const int incRowB,
     	Dtype *C, const int incRowC){
    
	    float alpha = 1;
	    float beta = 0;
	    float* alpha_ptr = &alpha;
	    float* beta_ptr = &beta;
	    err_t BLIS_INITIALIZED = bli_init();
		// assert(BLIS_INITIALIZED == BLIS_SUCCESS);    
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
}

#endif