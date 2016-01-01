#ifndef STRASSEN_MATRIX_MULTIPLICATION_H
#define STRASSEN_MATRIX_MULTIPLICATION_H

#include "setting.h"
#include "matrix_util.h"
#include "matrix_arithmetic.h"
#include "SimpleMatrixMultiplication.h"
#include <stdio.h>
#include <cblas.h>

void strassen_matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC);

void strassen_base_matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){
        return cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1, 
            A, k, 
            B, n, 
            0, 
            C, n);    
}

// M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
Dtype* strassen_make_M1_submatrix(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,    
    const Dtype *A_1_1, const int incRowA_1_1,
    const Dtype *A_2_2, const int incRowA_2_2,
    const Dtype *B_1_1, const int incRowB_1_1,
    const Dtype *B_2_2, const int incRowB_2_2){

    /*
    construct M1 by the formula
    M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
    */
    
    // T1 = (A_1_1 + A_2_2)
    Dtype* T1 = make_matrix(m, k);
    matrix_addition(m, k,
        A_1_1, incRowA_1_1,
        A_2_2, incRowA_2_2,
        T1, k);
   
    // T2 = (B_1_1 + B_2_2)
    Dtype* T2 = make_matrix(m, k);
    matrix_addition(m, k,
        B_1_1, incRowB_1_1,
        B_2_2, incRowB_2_2,
        T2, k);

    // M1 = T1 * T2
    Dtype* M1 = make_matrix(m, k);
    strassen_matrix_multiplication(
        m, n, k,
        T1, k,
        T2, k,
        M1, k);

    remove_matrix(T1);
    remove_matrix(T2);
    return M1;
}


Dtype* strassen_make_M2_submatrix(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,    
    const Dtype *A_2_1, const int incRowA_2_1,
    const Dtype *A_2_2, const int incRowA_2_2,
    const Dtype *B_1_1, const int incRowB_1_1){

    /*
    construct M2 by the formula
    M2 = (A_2_1 + A_2_2) * B_1_1
    */
    // T1 = A_2_1 + A_2_2
    Dtype* T1 = make_matrix(m, k);

    matrix_addition(m, k,
        A_2_1, incRowA_2_1,
        A_2_2, incRowA_2_2,
        T1, k);

    Dtype* M2 = make_matrix(m, k);
    strassen_matrix_multiplication(
        m, n, k,
        T1, k,
        B_1_1, incRowB_1_1,
        M2, k);

    remove_matrix(T1);
    return M2;
}


Dtype* strassen_make_M3_submatrix(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,    
    const Dtype *A_1_1, const int incRowA_1_1,
    const Dtype *B_1_2, const int incRowB_1_2,
    const Dtype *B_2_2, const int incRowB_2_2){

    /*
    construct M3 by the formula
    M3 = A_1_1 * (B_1_2 - B_2_2)
    */
    // T1 = B_1_2 - B_2_2
    Dtype* T1 = make_matrix(m, k);
    
    matrix_subtraction(m, k,
        B_1_2, incRowB_1_2,
        B_2_2, incRowB_2_2,
        T1, k);

    Dtype* M3 = make_matrix(m, k);
    
    strassen_matrix_multiplication(
        m, n, k,
        A_1_1, k,
        T1, k,
        M3, k);

    remove_matrix(T1);
    return M3;
}

Dtype* strassen_make_M4_submatrix(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,    
    const Dtype *A_2_2, const int incRowA_2_2,
    const Dtype *B_2_1, const int incRowB_2_1,
    const Dtype *B_1_1, const int incRowB_1_1){

    /*
    construct M4 by the formula
    M4 = A_2_2 * (B_2_1 - B_1_1)
    */
    Dtype* T1 = make_matrix(m, k);
    
    // T1 = B_2_1 - B_1_1
    matrix_subtraction(m, k,
        B_2_1, incRowB_2_1,
        B_1_1, incRowB_1_1,
        T1, k);

    Dtype* M4 = make_matrix(m, k);
    strassen_matrix_multiplication(
        m, n, k,
        A_2_2, k,
        T1, k,
        M4, k);

    remove_matrix(T1);
    return M4;
}

Dtype* strassen_make_M5_submatrix(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,    
    const Dtype *A_1_1, const int incRowA_1_1,
    const Dtype *A_1_2, const int incRowA_1_2,
    const Dtype *B_2_2, const int incRowB_2_2){

    /*
    construct M5 by the formula
    M5 = (A_1_1 + A_1_2) * B_2_2
    */
    // T1 = A_1_1 + A_1_2
    Dtype* T1 = make_matrix(m, k);
    matrix_addition(m, k,
        A_1_1, incRowA_1_1,
        A_1_2, incRowA_1_2,
        T1, k);

    Dtype* M5 = make_matrix(m, k);
    strassen_matrix_multiplication(
        m, n, k,
        T1, k,
        B_2_2, incRowB_2_2,
        M5, k);
    
    remove_matrix(T1);
    return M5;
}

// M6 = (A_2_1 - A_1_1) * (B_1_1 + B_1_2)
Dtype* strassen_make_M6_submatrix(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,    
    const Dtype *A_2_1, const int incRowA_2_1,
    const Dtype *A_1_1, const int incRowA_1_1,
    const Dtype *B_1_1, const int incRowB_1_1,
    const Dtype *B_1_2, const int incRowB_1_2){

    /*
    construct M6 by the formula
    M6 = (A_2_1 - A_1_1) * (B_1_1 + B_1_2)
    */    
    Dtype* T1 = make_matrix(m, k);
    Dtype* T2 = make_matrix(m, k);
    // T1 = (A_2_1 - A_1_1)
    matrix_subtraction(m, k,
        A_2_1, incRowA_2_1,
        A_1_1, incRowA_1_1,
        T1, k);
   
    // T2 = (B_1_1 + B_1_2)
    matrix_addition(m, k,
        B_1_1, incRowB_1_1,
        B_1_2, incRowB_1_2,
        T2, k);

    // M6 = T1 * T2
    Dtype* M6 = make_matrix(m, k);
    strassen_matrix_multiplication(
        m, n, k,
        T1, k,
        T2, k,
        M6, k);

    remove_matrix(T1);
    remove_matrix(T2);
    return M6;
}

// M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
Dtype* strassen_make_M7_submatrix(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,    
    const Dtype *A_1_2, const int incRowA_1_2,
    const Dtype *A_2_2, const int incRowA_2_2,
    const Dtype *B_2_1, const int incRowB_2_1,
    const Dtype *B_2_2, const int incRowB_2_2){

    /*
    construct M7 by the formula
    M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
    */
    Dtype* T1 = make_matrix(m, k);
    Dtype* T2 = make_matrix(m, k);    
    // T1 = (A_1_2 - A_2_2)
    matrix_subtraction(m, k,
        A_1_2, incRowA_1_2,
        A_2_2, incRowA_2_2,
        T1, k);
   
    // T2 = (B_2_1 + B_2_2)
    matrix_addition(m, k,
        B_2_1, incRowB_2_1,
        B_2_2, incRowB_2_2,
        T2, k);

    // M7 = T1 * T2
    Dtype* M7 = make_matrix(m, k);
    strassen_matrix_multiplication(
        m, n, k,
        T1, k,
        T2, k,
        M7, k);

    remove_matrix(T1);
    remove_matrix(T2);
    return M7;
}

//  C_1_1 = M1 + M4 - M5 + M7
void strassen_calculate_C_1_1(
    int m, int n, int k,
    const Dtype* M1, const Dtype* M4,
    const Dtype* M5, const Dtype* M7,
    Dtype* C_1_1, int incRowC_1_1
    ){
    // C_1_1 = M1 + M4 - M5 + M7
    
    // C_1_1 = M1 + M4
    matrix_addition(m, k,
        M1, k,
        M4, k,
        C_1_1, incRowC_1_1);
    
    // C_1_1 += M7
    matrix_addition(m, k,
        C_1_1, incRowC_1_1,
        M7, k,
        C_1_1, incRowC_1_1);
    
    // C_1_1 -= M5
    matrix_subtraction(m, k,
        C_1_1, incRowC_1_1,
        M5, k,
        C_1_1, incRowC_1_1);

}

void strassen_calculate_C_1_2(
    int m, int n, int k,
    const Dtype* M3, const Dtype* M5,
    Dtype* C_1_2, int incRowC_1_2
    ){

    matrix_addition(m, k,
        M3, k,
        M5, k,
        C_1_2, incRowC_1_2);

}

void strassen_calculate_C_2_1(
    int m, int n, int k,
    const Dtype* M2, const Dtype* M4,
    Dtype* C_2_1, int incRowC_2_1
    ){

    matrix_addition(m, k,
        M2, k,
        M4, k,
        C_2_1, incRowC_2_1);

}

//  C_2_2 = M1 - M2 + M3 + M6
void strassen_calculate_C_2_2(
    int m, int n, int k,
    const Dtype* M1, const Dtype* M2,
    const Dtype* M3, const Dtype* M6,
    Dtype* C_2_2, int incRowC_2_2
    ){

    // C_2_2 = M1 - M2
    matrix_subtraction(m, k,
        M1, k,
        M2, k,
        C_2_2, incRowC_2_2);
    
    // C_2_2 += M3
    matrix_addition(m, k,
        C_2_2, incRowC_2_2,
        M3, k,
        C_2_2, incRowC_2_2);
    // C_2_2 += M6

    matrix_addition(m, k,
        C_2_2, incRowC_2_2,
        M6, k,
        C_2_2, incRowC_2_2);

}

/* this script only works for square matrices 
   where the length is a power of 2
*/
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
       the base condition is defined as when all the dimensions are smaller than 256
	*/
	if(m <= 256 && n <= 256 && k <= 256){
        return strassen_base_matrix_multiplication(
            m, n, k,
            A, incRowA,
            B, incRowB,
            C, incRowC);
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
    Dtype* M1 = strassen_make_M1_submatrix(
            m1, n1, k1,    
            A_1_1, incRowA,
            A_2_2, incRowA,
            B_1_1, incRowB,
            B_2_2, incRowB);


    /*
	construct M2 by the formula
	M2 = (A_2_1 + A_2_2) * B_1_1
    */
    Dtype* M2 = strassen_make_M2_submatrix(
            m1, n1, k1,    
            A_2_1, incRowA,
            A_2_2, incRowA,
            B_1_1, incRowB);    

    /*
	construct M3 by the formula
	M3 = A_1_1 * (B_1_2 - B_2_2)
    */
    Dtype* M3 = strassen_make_M3_submatrix(
            m1, n1, k1,
            A_1_1, incRowA,
            B_1_2, incRowB,
            B_2_2, incRowB);

    /*
	construct M4 by the formula
	M4 = A_2_2 * (B_2_1 - B_1_1)
    */
    Dtype* M4 = strassen_make_M4_submatrix(
            m1, k1, k1,
            A_2_2, incRowA,
            B_2_1, incRowB,
            B_1_1, incRowB);


    /*
	construct M5 by the formula
	M5 = (A_1_1 + A_1_2) * B_2_2
    */
    Dtype* M5 = strassen_make_M5_submatrix(
            m1, n1, k1,
            A_1_1, incRowA,
            A_1_2, incRowA,
            B_2_2, incRowB);



	/*
	construct M6 by the formula
	M6 = (A_2_1 - A_1_1) * (B_1_1 + B_1_2)
	*/
    Dtype* M6 = strassen_make_M6_submatrix(
            m1, n1, k1,    
            A_2_1, incRowA,
            A_1_1, incRowA,
            B_1_1, incRowB,
            B_1_2, incRowB);

	/*
	construct M7 by the formula
	M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
	*/
    Dtype* M7 = strassen_make_M7_submatrix(
            m1, n1, k1,    
            A_1_2, incRowA,
            A_2_2, incRowA,
            B_2_1, incRowB,
            B_2_2, incRowB);

    /*
    compute C_1_1 by the formula
    C_1_1 = M1 + M4 - M5 + M7
    */
    strassen_calculate_C_1_1(
        m1, n1, k1,
        M1, M4, M5, M7,
        C_1_1, incRowC);

    /*
    compute C_1_2 by the formula
    C_1_2 = M3 + M5
    */
    strassen_calculate_C_1_2(
        m1, n1, k1,
        M3, M5,
        C_1_2, incRowC);

    /*
    compute C_2_1 by the formula
    C_2_1 = M2 + M4
    */
    strassen_calculate_C_2_1(
        m1, n1, k1,
        M2, M4,
        C_2_1, incRowC);

    /*
    compute C_2_2 by the formula
    C_2_2 = M1 - M2 + M3 + M6
    */
    strassen_calculate_C_2_2(
        m1, n1, k1,
        M1, M2, M3, M6,
        C_2_2, incRowC);

    /*
    remove the working space
    */
    remove_matrix(M1);
    remove_matrix(M2);
    remove_matrix(M3);
    remove_matrix(M4);
    remove_matrix(M5);
    remove_matrix(M6);
    remove_matrix(M7);
}

#endif