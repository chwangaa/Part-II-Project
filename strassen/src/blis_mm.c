#include "matrix.h"
#include "blis.h"

void blis_mm(
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