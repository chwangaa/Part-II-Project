#ifndef STRASSEN_UTIL_H
#define STRASSEN_UTIL_H
#include "../setting.h"
#include "matrix_util.h"
#include "Cblas_GEMM.h"
#include "SimpleMatrixMultiplication.h"
#include <stdbool.h>

void strassen_base_matrix_multiplication(
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

void matrix_partial_addition(Dtype* result, int rM, int rN, int rincRow,
							 const Dtype* adder,  int aM, int aN, int aincRow){
	debug_assert(aM <= rM);
	debug_assert(aN <= rN);
	for(int i = 0; i < aM; i++){
		// Dtype* r = &result[i*rincRow];
		// Dtype* a = &adder[i*aincRow];
		for(int j = 0; j < aN; j++){
			result[i*rincRow + j] += adder[i*aincRow + j];
			//*r += *a;
			//r++;
			//a++;
		}
	}
}

void matrix_partial_subtraction(Dtype* result, int rM, int rN, int rincRow,
							 const Dtype* adder,  int aM, int aN, int aincRow){
	debug_assert(aM <= rM);
	debug_assert(aN <= rN);
	for(int i = 0; i < aM; i++){
		for(int j = 0; j < aN; j++){
			result[i*rincRow + j] -= adder[i*aincRow + j];
		}
	}
}

Dtype* addDifferentSizedMatrix(
	Dtype* const Larger, int lm, int ln, int incRowL,
	Dtype* const Smaller, int sm, int sn, int incRowS){

	Dtype* new_matrix = matrix_copy(Larger, lm, ln, incRowL);
	matrix_partial_addition(new_matrix, lm, ln, ln,
							Smaller, sm, sn, incRowS);
	return new_matrix;
}

Dtype* subtractDifferentSizedMatrix(
	Dtype* const Larger, int lm, int ln, int incRowL,
	Dtype* const Smaller, int sm, int sn, int incRowS){

	Dtype* new_matrix = matrix_copy(Larger, lm, ln, incRowL);
	matrix_partial_subtraction(new_matrix, lm, ln, ln,
							Smaller, sm, sn, incRowS);
	return new_matrix;
}

int getNumberLargerThanXAndIsPowerOfTwo(unsigned int X){
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

int strassenCalculateNewSize(unsigned int X){
	int result = 1;
	int value = X;
	// if X is already smaller than the limit, then return X
	if(X <= limit_X){
		return X;
	}

	while(X > limit_X){
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
int max(int a, int b){
	if(a > b){
		return a;
	}
	else{
		return b;
	}
}

int maxThree(int a, int b, int c){
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

bool sizeOkay(const unsigned int X){
    bool isEven = false;
    bool isLarge = false;

    if(X % 2 == 0){
        isEven = true;
    }
    if(X > limit_X){
        isLarge = true;
    }
    return (isEven || isLarge);
}

bool sizeSmallerThanLimit(const unsigned int X){
    if(X < limit_X){
        return true;
    }
    else{
        return false;
    }
}

bool baseConditionReached(const unsigned int m,
                          const unsigned int n,
                          const unsigned int k){

    if(sizeSmallerThanLimit(m) || sizeSmallerThanLimit(n) || sizeSmallerThanLimit(k)){
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

Dtype* padMatrixToPowerSquareMatrix(Dtype* old_matrix, const unsigned int M, const unsigned int N,
									const unsigned incRow){
	int side_length = max(M, N);
	int new_length = getNumberLargerThanXAndIsPowerOfTwo(side_length);
	return pad_matrix(old_matrix, M, N, incRow, new_length, new_length);
}

Dtype* padMatrixToStrassenMatrix(Dtype* old_matrix, const unsigned int M, const unsigned int N,
								 const unsigned int incRow){
	int newM = strassenCalculateNewSize(M);
	int newN = strassenCalculateNewSize(N);
	return pad_matrix(old_matrix, M, N, incRow, newM, newN);
}

#endif