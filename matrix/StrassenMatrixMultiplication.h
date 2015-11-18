#ifndef STRASSEN_MATRIX_MULTIPLICATION_H
#define STRASSEN_MATRIX_MULTIPLICATION_H

#include "setting.h"
#include "matrix_util.h"
#include "matrix_arithmetic.h"
#include "SimpleMatrixMultiplication.h"
#include <stdio.h>
#include <cblas.h>
// this script only works for square matrices where the length is a power of 2
void strassen_matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

	// the matrices must have positive dimension
	assert(m > 0);
	assert(n > 0);
	assert(k > 0);
	
	/* check if the base case has reached
	   here we recurse until all the dimension are smaller than 2
	*/
	if(m <= 256 && n <= 256 && k <= 256){
		// return SimpleMatrixMultiplication(
		// 	m, n, k,
		// 	A, incRowA,
		// 	B, incRowB,
		// 	C, incRowC);
        return cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1, 
            A, k, 
            B, n, 
            0, 
            C, n);
	}


	/*
	We divide A, B, C into subsections as the following:
	A = |A_1_1, A_1_2|   B = |B_1_1, B_1_2|   C = |C_1_1, C_1_2|
	    |A_2_1, A_2_2|       |B_2_1, B_2_2|       |C_2_1, C_2_2|


	We first calculate 7 temporary matrices as the follow:
	M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
	M2 = (A_2_1 + A_2_2) * B_1_1
	M3 = A_1_1 * (B_1_2 - B_2_2)
	M4 = A_2_2 * (B_2_1 - B_1_1)
	M5 = (A_1_1 + A_1_2) * B_2_2
	M6 = (A_2_1 - A_1_1) * (B_1_1 + B_1_2)
	M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)

	Then, we compute C section by section according to:
	C_1_1 = M1 + M4 - M5 + M7
	C_1_2 = M3 + M5
	C_2_1 = M2 + M4
	C_2_2 = M1 - M2 + M3 + M6 
	*/
	const unsigned int m1 = m / 2;
	const unsigned int m2 = m - m1;	
	const unsigned int n1 = n / 2;
	const unsigned int n2 = n - n1;
	const unsigned int k1 = k / 2;
	const unsigned int k2 = k - k1;
	
	const Dtype* A_1_1 = A;
	const Dtype* A_1_2 = A_1_1 + k1;
	const Dtype* A_2_1 = A_1_1 + incRowA*m1;
	const Dtype* A_2_2 = A_2_1 + k1;

	Dtype* C_1_1 = C;
	Dtype* C_1_2 = C_1_1 + n1;
	Dtype* C_2_1 = C_1_1 + incRowC*m1;
	Dtype* C_2_2 = C_2_1 + n1;

	const Dtype* B_1_1 = B;
	const Dtype* B_1_2 = B_1_1 + n1;
	const Dtype* B_2_1 = B_1_1 + incRowB*k1;
	const Dtype* B_2_2 = B_2_1 + n1;

	// first construct the temporary for A_1_1 + A_2_2
	// this version assumes square matrices, m1 == n1 == k1
	assert(m1 == k1);
	assert(m1 == n1);
	assert(m1 == m2);
	/*
	construct M1 by the formula
	M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
	*/
	
	// T1 = (A_1_1 + A_2_2)
	Dtype* T1 = make_matrix(m1, k1);
	matrix_addition(m1, k1,
		A_1_1, incRowA,
		A_2_2, incRowA,
		T1, k1);
   
    // T2 = (B_1_1 + B_2_2)
    Dtype* T2 = make_matrix(m1, k1);
    matrix_addition(m1, k1,
    	B_1_1, incRowB,
    	B_2_2, incRowB,
    	T2, k1);

    // M1 = T1 * T2
    Dtype* M1 = make_matrix(m1, k1);
    strassen_matrix_multiplication(
    	m1, n1, k1,
    	T1, k1,
    	T2, k1,
    	M1, k1);

    // now that T1, T2 can hold new temporaries
    /*
	construct M2 by the formula
	M2 = (A_2_1 + A_2_2) * B_1_1
    */
    // T1 = A_2_1 + A_2_2
    matrix_addition(m1, k1,
    	A_2_1, incRowA,
    	A_2_2, incRowA,
    	T1, k1);

    Dtype* M2 = make_matrix(m1, k1);
    strassen_matrix_multiplication(
    	m1, n1, k1,
    	T1, k1,
    	B_1_1, incRowB,
    	M2, k1);

    /*
	construct M3 by the formula
	M3 = A_1_1 * (B_1_2 - B_2_2)
    */
    // T1 = B_1_2 - B_2_2
    matrix_subtraction(m1, k1,
    	B_1_2, incRowB,
    	B_2_2, incRowB,
    	T1, k1);

    Dtype* M3 = make_matrix(m1, k1);
    strassen_matrix_multiplication(
    	m1, n1, k1,
    	A_1_1, k1,
    	T1, k1,
    	M3, k1);

    /*
	construct M4 by the formula
	M4 = A_2_2 * (B_2_1 - B_1_1)
    */
    // T1 = B_2_1 - B_1_1
    matrix_subtraction(m1, k1,
    	B_2_1, incRowB,
    	B_1_1, incRowB,
    	T1, k1);

    Dtype* M4 = make_matrix(m1, k1);
    strassen_matrix_multiplication(
    	m1, n1, k1,
    	A_2_2, k1,
    	T1, k1,
    	M4, k1);


    /*
	construct M5 by the formula
	M5 = (A_1_1 + A_1_2) * B_2_2
    */
    // T1 = A_1_1 + A_1_2
    matrix_addition(m1, k1,
    	A_1_1, incRowA,
    	A_1_2, incRowA,
    	T1, k1);

    Dtype* M5 = make_matrix(m1, k1);
    strassen_matrix_multiplication(
    	m1, n1, k1,
    	T1, k1,
    	B_2_2, incRowB,
    	M5, k1);


	/*
	construct M6 by the formula
	M6 = (A_2_1 - A_1_1) * (B_1_1 + B_1_2)
	*/
	
	// T1 = (A_2_1 - A_1_1)
	matrix_subtraction(m1, k1,
		A_2_1, incRowA,
		A_1_1, incRowA,
		T1, k1);
   
    // T2 = (B_1_1 + B_1_2)
    matrix_addition(m1, k1,
    	B_1_1, incRowB,
    	B_1_2, incRowB,
    	T2, k1);

    // M6 = T1 * T2
    Dtype* M6 = make_matrix(m1, k1);
    strassen_matrix_multiplication(
    	m1, n1, k1,
    	T1, k1,
    	T2, k1,
    	M6, k1);

	/*
	construct M7 by the formula
	M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
	*/
	
	// T1 = (A_1_2 - A_2_2)
	matrix_subtraction(m1, k1,
		A_1_2, incRowA,
		A_2_2, incRowA,
		T1, k1);
   
    // T2 = (B_2_1 + B_2_2)
    matrix_addition(m1, k1,
    	B_2_1, incRowB,
    	B_2_2, incRowB,
    	T2, k1);

    // M7 = T1 * T2
    Dtype* M7 = make_matrix(m1, k1);
    strassen_matrix_multiplication(
    	m1, n1, k1,
    	T1, k1,
    	T2, k1,
    	M7, k1);

    /*
    compute C_1_1 by the formula
    C_1_1 = M1 + M4 - M5 + M7
    */
    // C_1_1 = M1 + M4
    matrix_addition(m1, k1,
    	M1, k1,
    	M4, k1,
    	C_1_1, incRowC);
    // C_1_1 += M7
    matrix_addition(m1, k1,
    	C_1_1, incRowC,
    	M7, k1,
    	C_1_1, incRowC);
    // C_1_1 -= M5
    matrix_subtraction(m1, k1,
    	C_1_1, incRowC,
    	M5, k1,
    	C_1_1, incRowC);

    /*
    compute C_1_2 by the formula
    C_1_2 = M3 + M5
    */
    matrix_addition(m1, k1,
    	M3, k1,
    	M5, k1,
    	C_1_2, incRowC);

    /*
    compute C_2_1 by the formula
    C_2_1 = M2 + M4
    */
    matrix_addition(m1, k1,
    	M2, k1,
    	M4, k1,
    	C_2_1, incRowC);

    /*
    compute C_2_2 by the formula
    C_2_2 = M1 - M2 + M3 + M6
    */
    // C_2_2 = M1 - M2
    matrix_subtraction(m1, k1,
    	M1, k1,
    	M2, k1,
    	C_2_2, incRowC);
    
    // C_2_2 += M3
    matrix_addition(m1, k1,
    	C_2_2, incRowC,
    	M3, k1,
    	C_2_2, incRowC);
    // C_2_2 += M6
    matrix_addition(m1, k1,
    	C_2_2, incRowC,
    	M6, k1,
    	C_2_2, incRowC);

    /*
    remove the working space
    */
    remove_matrix(T1);
    remove_matrix(T2);
    remove_matrix(M1);
    remove_matrix(M2);
    remove_matrix(M3);
    remove_matrix(M4);
    remove_matrix(M5);
    remove_matrix(M6);
    remove_matrix(M7);

}

#endif