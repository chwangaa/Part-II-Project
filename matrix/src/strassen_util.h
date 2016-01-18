#ifndef STRASSEN_UTIL_H
#define STRASSEN_UTIL_H
#include "../setting.h"
#include "matrix_util.h"
#include "SimpleMatrixMultiplication.h"
#include <stdbool.h>
#include "Cblas_GEMM.h"
#include "Cblas_packed.h"

static void strassen_base_matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){
        #ifdef GEMM
	        return cblas_gemm(
	            m, n, k,
	            A, incRowA, 
	            B, incRowB, 
	            C, incRowC);  
		#else        
	        return SimpleMatrixMultiplication(
	            m, n, k,
	            A, incRowA,
	            B, incRowB,
	            C, incRowC);  
        #endif
}


static void packed_strassen_base_matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){
        #ifdef GEMM
	        return packed_cblas_gemm(
	            m, n, k,
	            A, incRowA, 
	            B, incRowB, 
	            C, incRowC);  
		#else        
	        assert(0);  
        #endif
}


static int getNumberLargerThanXAndIsPowerOfTwo(unsigned int X){
	int result = 1;
	int value = X;
	while(X > 1){
		X /= 2;
		result *= 2;
	}
	if(result >= value){
		return result;
	}
	else{
		return result * 2;
	}
}

static int strassenCalculateNewSize(unsigned int X){
	int result = 1;
	int value = X;
	// if X is already smaller than the limit, then return X
	if(X <= limit_M){
		return X;
	}

	while(X > limit_M){
		int rem = X % 2; 
		X = X / 2 + rem;
		result *= 2;
	}
	result *= X;
	if(result >= value){
		return result;
	}
	else{
		return result * 2;
	}	

}

static int max(int a, int b){
	if(a > b){
		return a;
	}
	else{
		return b;
	}
}

static int maxThree(int a, int b, int c){
	int max = -100000;
	if(a > max){
		max = a;
	}
	if(b > max){
		max = b;
	}
	if(c > max){
		max = c;
	}
	return max;
}



static bool baseConditionReached(const unsigned int m,
                          const unsigned int n,
                          const unsigned int k){

    if(m <= limit_M || n <= limit_N || k < limit_K){
        return true;
    }
    else{
    	return false;
    }
    // // if all dimensions are larger than the limit, then clearly the condition is not yet reached
    // if(sizeOkay(m) && sizeOkay(n) && sizeOkay(k)){
    //     return false;
    // }
    // else{
    //     return true;
    // }
}

static bool packedbaseConditionReached(const unsigned int m,
                          const unsigned int n,
                          const unsigned int k){

    if(m % 16 == 0 && n % 16 ==0 && k > limit_K && m > limit_M && n > limit_N){
        return false;
    }
    else{
    	return true;
    }
}

static Dtype* padMatrixToPowerSquareMatrix(Dtype* old_matrix, const unsigned int M, const unsigned int N,
									const unsigned incRow){
	int side_length = max(M, N);
	int new_length = getNumberLargerThanXAndIsPowerOfTwo(side_length);
	return pad_matrix(old_matrix, M, N, incRow, new_length, new_length);
}

static Dtype* padMatrixToStrassenMatrix(Dtype* old_matrix, const unsigned int M, const unsigned int N,
								 const unsigned int incRow){
	int newM = strassenCalculateNewSize(M);
	int newN = strassenCalculateNewSize(N);
	return pad_matrix(old_matrix, M, N, incRow, newM, newN);
}

#endif