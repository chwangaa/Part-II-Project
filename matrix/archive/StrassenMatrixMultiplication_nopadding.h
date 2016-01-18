/*
 * 2016 Jan 03
 * Chihang Wang
 * in this version, the matrix is not padded at all
 * however, the temporary matrices in each computation of
 * submatrix is first set to be large enough, hence the trick
*/
 
#ifndef STRASSEN_MATRIX_MULTIPLICATION_H
#define STRASSEN_MATRIX_MULTIPLICATION_H

#include "setting.h"
#include "strassen_util.h"
#include "matrix_arithmetic.h"
#include "SimpleMatrixMultiplication.h"
#include <stdio.h>
#include <cblas.h>

void strassen_matrix_multiplication_worker(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC);

// M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
Dtype* strassen_make_M1_submatrix(    
    const Dtype *A_1_1, int a11m, int a11n, const int incRowA_1_1,
    const Dtype *A_2_2, int a22m, int a22n, const int incRowA_2_2,
    const Dtype *B_1_1, int b11m, int b11n, const int incRowB_1_1,
    const Dtype *B_2_2, int b22m, int b22n, const int incRowB_2_2){

    /*
    construct M1 by the formula
    M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
    */
    // T1 = (A_1_1 + A_2_2)
    Dtype* T1 = make_matrix(a11m, a11n);
    

    // T1 = A_1_1
    matrix_copyTo(A_1_1, a11m, a11n, incRowA_1_1,
                  T1, a11m, a11n, a11n);

    // T1 += A_2_2
    matrix_partial_addition(
        T1, a11m, a11n, a11n,
        A_2_2, a22m, a22n, incRowA_2_2
        );
   
    // T2 = (B_1_1 + B_2_2)

    // T2 = B_1_1
    Dtype* T2 = make_matrix(b11m, b11n);

    matrix_copyTo(B_1_1, b11m, b11n, incRowB_1_1,
                  T2, b11m, b11n, b11n);
    // T2 += B_2_2
    matrix_partial_addition(
        T2, b11m, b11n, b11n,
        B_2_2, b22m, b22n, incRowB_2_2
        );

    // M1 = T1 * T2
    int m = a11m;
    int n = b11n;
    int k = a11n;
    Dtype* M1 = make_matrix(m, n);
    strassen_matrix_multiplication_worker(
        m, n, k,
        T1, k,
        T2, n,
        M1, n);
    remove_matrix(T1);
    remove_matrix(T2);
    return M1;
}


Dtype* strassen_make_M2_submatrix(
    const Dtype *A_2_1, int a21m, int a21n, const int incRowA_2_1,
    const Dtype *A_2_2, int a22m, int a22n, const int incRowA_2_2,
    const Dtype *B_1_1, int b11m, int b11n, const int incRowB_1_1){

    // sanity check
    debug_assert(a21m == a22m);
    debug_assert(a21n == b11m);
    debug_assert(a21n >= a22n);

    int m2 = a21m;
    int k1 = a21n;
    int n1 = b11n;
    /*
    construct M2 by the formula
    M2 = (A_2_1 + A_2_2) * B_1_1
    */
    // T1 = A_2_1 + A_2_2

    Dtype* T1 = make_matrix(m2, k1);
    // T1 = A_2_1
    matrix_copyTo(A_2_1, m2, k1, incRowA_2_1,
                  T1, m2, k1, k1);
    // T1 += A_2_2
    matrix_partial_addition(
        T1, m2, k1, k1,
        A_2_2, a22m, a22n, incRowA_2_2
        );

    Dtype* M2 = make_matrix(m2, n1);
    strassen_matrix_multiplication_worker                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (
        m2, n1, k1,
        T1, k1,
        B_1_1, incRowB_1_1,
        M2, n1);

    remove_matrix(T1);
    return M2;
}


Dtype* strassen_make_M3_submatrix(
    const Dtype *A_1_1, int a11m, int a11n, const int incRowA_1_1,
    const Dtype *B_1_2, int b12m, int b12n, const int incRowB_1_2,
    const Dtype *B_2_2, int b22m, int b22n, const int incRowB_2_2){


    // sanity check
    debug_assert(b12n == b22n);
    debug_assert(a11n == b12m);
    int m1 = a11m;
    int k1 = b12m;
    int n1 = b12n;

    /*
    construct M3 by the formula
    M3 = A_1_1 * (B_1_2 - B_2_2)
    */
    Dtype* T1 = make_matrix(k1, n1);
    // T1 = B_1_2
    matrix_copyTo(B_1_2, b12m, b12n, incRowB_1_2,
                  T1, k1, n1, n1);
    // T1 -= B_2_2
    matrix_partial_subtraction(
        T1, k1, n1, n1,
        B_2_2, b22m, b22n, incRowB_2_2
        );

    Dtype* M3 = make_matrix(m1, n1);
    
    strassen_matrix_multiplication_worker(
        m1, n1, k1,
        A_1_1, incRowA_1_1,
        T1, n1,
        M3, n1); 

    remove_matrix(T1);
    return M3;
}

Dtype* strassen_make_M4_submatrix(    
    const Dtype *A_2_2, int a22m, int a22n, const int incRowA_2_2,
    const Dtype *B_2_1, int b21m, int b21n, const int incRowB_2_1,
    const Dtype *B_1_1, int b11m, int b11n, const int incRowB_1_1){

    // sanity check
    debug_assert(b21n == b11n);
    debug_assert(a22n <= b21m);
    int k1 = b11m;
    int n1 = b11n;
    int m2 = a22m;
    /*
    construct M4 by the formula
    M4 = A_2_2 * (B_1_1 - B_2_1)
    */
    Dtype* T1 = make_matrix(k1, n1);
    
    // T1 = B_1_1
    matrix_copyTo(B_1_1, b11m, b11n, incRowB_1_1,
                  T1, k1, n1, n1);
    // T1 -= B_2_1
    matrix_partial_subtraction(
        T1, k1, n1, n1,
        B_2_1, b21m, b21n, incRowB_2_1
        );

    Dtype* T2;
    bool new_T2 = false;
    int incRowT2 = incRowA_2_2;
    if(a22n == k1){
        T2 = A_2_2;
    }
    else{
        new_T2 = true;
        T2 = pad_matrix(A_2_2, a22m, a22n, incRowA_2_2,
                   m2, k1);
        incRowT2 = k1;
    }

    Dtype* M4 = make_matrix(m2, n1);
    strassen_matrix_multiplication_worker(
        m2, n1, k1,
        T2, incRowT2,
        T1, n1,
        M4, n1);

    remove_matrix(T1);
    if(new_T2){
        remove_matrix(T2);
    }
    return M4;
}

Dtype* strassen_make_M5_submatrix(
    const Dtype *A_1_1, int a11m, int a11n, const int incRowA_1_1,
    const Dtype *A_1_2, int a12m, int a12n, const int incRowA_1_2,
    const Dtype *B_2_2, int b22m, int b22n, const int incRowB_2_2){

    // sanity check
    debug_assert(a11m == a12m);
    debug_assert(a11n >= a12n);
    debug_assert(a11n >= b22m);
    int m1 = a11m;
    int k1 = a11n;
    int n2 = b22n;
    /*
    construct M5 by the formula
    M5 = (A_1_1 + A_1_2) * B_2_2
    */
    // T1 = A_1_1 + A_1_2
    Dtype* T1 = make_matrix(m1, k1);
    // T1 = A_1_1
    matrix_copyTo(A_1_1, a11m, a11n, incRowA_1_1,
                  T1, m1, k1, k1);
    // T1 += A_1_2
    matrix_partial_addition(
        T1, m1, k1, k1,
        A_1_2, a12m, a12n, incRowA_1_2
        );

    Dtype* T2;
    bool new_T2 = false;
    int incRowT2 = incRowB_2_2;
    if(b22m == k1){
        T2 = B_2_2;
    }
    else{
        T2 = pad_matrix(B_2_2, b22m, b22n, incRowB_2_2,
                        k1, n2);
        new_T2 = true;
        incRowT2 = n2;
    }

    Dtype* M5 = make_matrix(m1, n2);
    strassen_matrix_multiplication_worker(
        m1, n2, k1,
        T1, k1,
        T2, incRowT2,
        M5, n2);
    
    remove_matrix(T1);
    if(new_T2){
        remove_matrix(T2);
    }
    return M5;
}

// M6 = (A_1_1 - A_2_1) * (B_1_1 + B_1_2)
Dtype* strassen_make_M6_submatrix(    
    const Dtype *A_2_1, int a21m, int a21n, const int incRowA_2_1,
    const Dtype *A_1_1, int a11m, int a11n, const int incRowA_1_1,
    const Dtype *B_1_1, int b11m, int b11n, const int incRowB_1_1,
    const Dtype *B_1_2, int b12m, int b12n, const int incRowB_1_2){

    // sanity check
    debug_assert(a11m >= a21m);
    debug_assert(a11n == a21n);
    debug_assert(b11m == b12m);
    debug_assert(b11n >= b12n);
    debug_assert(a11n == b11m);
    int m1 = a11m;
    int k1 = a11n;
    int n1 = b11n;
    /*
    construct M6 by the formula
    M6 = (A_1_1 - A_2_1) * (B_1_1 + B_1_2)
    */    
    Dtype* T1 = make_matrix(m1, k1);
    Dtype* T2 = make_matrix(k1, n1);
    // T1 = (A_1_1 - A_2_1)
    matrix_copyTo(A_1_1, a11m, a11n, incRowA_1_1,
                  T1, m1, k1, k1);
    matrix_partial_subtraction(T1, m1, k1, k1,
                  A_2_1, a21m, a21n, incRowA_2_1);
   
    // T2 = (B_1_1 + B_1_2)
    matrix_copyTo(B_1_1, b11m, b11n, incRowB_1_1,
                  T2, k1, n1, n1);
    matrix_partial_addition(T2, k1, n1, n1,
                  B_1_2, b12m, b12n, incRowB_1_2);

    // M6 = T1 * T2
    Dtype* M6 = make_matrix(m1, n1);
    strassen_matrix_multiplication_worker(
        m1, n1, k1,
        T1, k1,
        T2, n1,
        M6, n1);

    remove_matrix(T1);
    remove_matrix(T2);
    return M6;
}

// M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
Dtype* strassen_make_M7_submatrix(
    const Dtype *A_1_2, int a12m, int a12n, const int incRowA_1_2,
    const Dtype *A_2_2, int a22m, int a22n, const int incRowA_2_2,
    const Dtype *B_2_1, int b21m, int b21n, const int incRowB_2_1,
    const Dtype *B_2_2, int b22m, int b22n, const int incRowB_2_2){


    // sanity check
    debug_assert(a12m >= a22m);
    debug_assert(a12n == a22n);
    debug_assert(b21m == b22m);
    debug_assert(b21n >= b22n);
    debug_assert(a12n == b21m);
    
    int m1 = a12m;
    int n1 = b21n;
    int k2 = a12n;
    /*
    construct M7 by the formula
    M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
    */
    Dtype* T1 = make_matrix(m1, k2);
    Dtype* T2 = make_matrix(k2, n1);    
    // T1 = (A_1_2 - A_2_2)
    matrix_copyTo(A_1_2, a12m, a12n, incRowA_1_2,
                  T1, m1, k2, k2);
    matrix_partial_subtraction(T1, m1, k2, k2,
                  A_2_2, a22m, a22n, incRowA_2_2);   
    // T2 = (B_2_1 + B_2_2)
    matrix_copyTo(B_2_1, b21m, b21n, incRowB_2_1,
                  T2, k2, n1, n1);
    matrix_partial_addition(T2, k2, n1, n1,
                  B_2_2, b22m, b22n, incRowB_2_2);
        
    // M7 = T1 * T2
    Dtype* M7 = make_matrix(m1, n1);
    strassen_matrix_multiplication_worker(
        m1, n1, k2,
        T1, k2,
        T2, n1,
        M7, n1);
    remove_matrix(T1);
    remove_matrix(T2);
    return M7;
}

//  C_1_1 = M1 - M4 - M5 + M7
void strassen_calculate_C_1_1(
    const Dtype* M1, int m1m, int m1n,
    const Dtype* M4, int m4m, int m4n,
    const Dtype* M5, int m5m, int m5n,
    const Dtype* M7, int m7m, int m7n,
    Dtype* C_1_1,    int M,   int N,
    int incRowC_1_1
    ){
    // C_1_1 = M1 - M4 - M5 + M7
    
    // C_1_1 = M1
    matrix_copyTo(M1, m1m, m1n, m1n,
                  C_1_1, M, N, incRowC_1_1);
    // C_1_1 -= M4
    matrix_partial_subtraction(C_1_1, M, N, incRowC_1_1,
                  M4, m4m, m4n, m4n);
    // C_1_1 += M7
    matrix_partial_addition(C_1_1, M, N, incRowC_1_1,
                  M7, m7m, m7n, m7n);
    // C_1_1 -= M5
    matrix_partial_subtraction(C_1_1, M, N, incRowC_1_1,
                  M5, m5m, m5n, m5n);
}

// C_1_2 = M3 + M5
void strassen_calculate_C_1_2(
    const Dtype* M3, int m3m, int m3n, 
    const Dtype* M5, int m5m, int m5n,
    Dtype* C_1_2, int M, int N,
    int incRowC_1_2
    ){

    matrix_copyTo(M3, m3m, m3n, m3n,
                  C_1_2, M, N, incRowC_1_2);
    matrix_partial_addition(C_1_2, M, N, incRowC_1_2,
                  M5, m5m, m5n, m5n);

}

// C_2_1 = M2 - M4
void strassen_calculate_C_2_1(
    const Dtype* M2, int m2m, int m2n,
    const Dtype* M4, int m4m, int m4n,
    Dtype* C_2_1, int M, int N,
    int incRowC_2_1
    ){

    matrix_copyTo(M2, m2m, m2n, m2n,
                  C_2_1, M, N, incRowC_2_1);

    matrix_partial_subtraction(C_2_1, M, N, incRowC_2_1,
                  M4, m4m, m4n, m4n);
}

//  C_2_2 = M1 - M2 + M3 - M6
void strassen_calculate_C_2_2(
    const Dtype* M1, int m1m, int m1n,
    const Dtype* M2, int m2m, int m2n,
    const Dtype* M3, int m3m, int m3n,
    const Dtype* M6, int m6m, int m6n,
    Dtype* C_2_2,    int M,   int N,
    int incRowC_2_2
    ){

    // M1 = M1 - M2

    matrix_partial_subtraction(M1, m1m, m1n, m1n,
                  M2, m2m, m2n, m2n);
    
    // M1 += M3
    matrix_partial_addition(M1, m1m, m1n, m1n,
                  M3, m3m, m3n, m3n);
    
    // M1 -= M6
    matrix_partial_subtraction(M1, m1m, m1n, m1n,
                  M6, m6m, m6n, m6n);

    // C_2_2 = M1
    matrix_copyTo(M1, m1m, m1n, m1n,
                  C_2_2, M, N, incRowC_2_2);
}


void strassen_matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

    // int newM = strassenCalculateNewSize(m);
    // int newN = strassenCalculateNewSize(n);
    // int newK = strassenCalculateNewSize(k);
    // Dtype* newA = pad_matrix(A, m, k, incRowA, newM, newK);
    // Dtype* newB = pad_matrix(B, k, n, incRowB, newK, newN);
    // Dtype* newC = pad_matrix(C, m, n, incRowC, newM, newN);

    return strassen_matrix_multiplication_worker(
        m, n, k,
        A, incRowA,
        B, incRowB,
        C, incRowC);

    // matrix_copyTo(newC, newM, newN, newN,
    //               C, m, n, incRowC);

    // // remove the extra workspace
    // remove_matrix(newA);
    // remove_matrix(newB);
    // remove_matrix(newC);

}
/* this script only works for square matrices 
   where the length is a power of 2
*/
void strassen_matrix_multiplication_worker(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

    // the matrices must have positive dimension
    debug_assert(m > 0);
    debug_assert(n > 0);
    debug_assert(k > 0);
    
    /* check if the base case has reached
       the base condition is defined as when all the dimensions are smaller than 256
    */
    // fprintf(stderr, "multiplying dimension m(%d), n(%d), k(%d) \n", m, n, k);
    if(baseConditionReached(m, n, k)){
        strassen_base_matrix_multiplication(
            m, n, k,
            A, incRowA,
            B, incRowB,
            C, incRowC);
        return;
    }
    /*
    We divide A, B, C into subsections as the following:
    A = |A_1_1, A_1_2|   B = |B_1_1, B_1_2|   C = |C_1_1, C_1_2|
        |A_2_1, A_2_2|       |B_2_1, B_2_2|       |C_2_1, C_2_2|


    We first calculate 7 temporary matrices as the follow:
    M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
    M2 = (A_2_1 + A_2_2) * B_1_1
    M3 = A_1_1 * (B_1_2 - B_2_2)
    M4 = A_2_2 * (B_1_1 - B_1_2)
    M5 = (A_1_1 + A_1_2) * B_2_2
    M6 = (A_1_1 - A_1_2) * (B_1_1 + B_1_2)
    M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)

    Then, we compute C section by section according to:
    C_1_1 = M1 - M4 - M5 + M7
    C_1_2 = M3 + M5
    C_2_1 = M2 - M4
    C_2_2 = M1 - M2 + M3 - M6 
    */
    const unsigned int m2 = m / 2;
    const unsigned int m1 = m - m2; 
    const unsigned int n2 = n / 2;
    const unsigned int n1 = n - n2;
    const unsigned int k2 = k / 2;
    const unsigned int k1 = k - k2;
    
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

    // this version assumes the dimension is divisible by 2

    // first construct the temporary for A_1_1 + A_2_2
    /*
    construct M1 by the formula
    M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
    */    

    Dtype* M1 = strassen_make_M1_submatrix(    
            A_1_1, m1, k1, incRowA,
            A_2_2, m2, k2, incRowA,
            B_1_1, k1, n1, incRowB,
            B_2_2, k2, n2, incRowB);

    /*
    construct M2 by the formula
    M2 = (A_2_1 + A_2_2) * B_1_1
    */
    Dtype* M2 = strassen_make_M2_submatrix(
            A_2_1, m2, k1, incRowA,
            A_2_2, m2, k2, incRowA,
            B_1_1, k1, n1, incRowB);    

    /*
    construct M3 by the formula
    M3 = A_1_1 * (B_1_2 - B_2_2)
    */
    Dtype* M3 = strassen_make_M3_submatrix(
            A_1_1, m1, k1, incRowA,
            B_1_2, k1, n2, incRowB,
            B_2_2, k2, n2, incRowB);

    /*
    construct M4 by the formula
    M4 = A_2_2 * (B_2_1 - B_1_1)
    */
    Dtype* M4 = strassen_make_M4_submatrix(
            A_2_2, m2, k2, incRowA,
            B_2_1, k2, n1, incRowB,
            B_1_1, k1, n1, incRowB);

    /*
    construct M5 by the formula
    M5 = (A_1_1 + A_1_2) * B_2_2
    */
    Dtype* M5 = strassen_make_M5_submatrix(
            A_1_1, m1, k1, incRowA,
            A_1_2, m1, k2, incRowA,
            B_2_2, k2, n2, incRowB);

    /*
    construct M6 by the formula
    M6 = (A_1_1 - A_2_1) * (B_1_1 + B_1_2)
    */
    Dtype* M6 = strassen_make_M6_submatrix(
            A_2_1, m2, k1, incRowA,
            A_1_1, m1, k1, incRowA,
            B_1_1, k1, n1, incRowB,
            B_1_2, k1, n2, incRowB);

    /*
    construct M7 by the formula
    M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
    */
    Dtype* M7 = strassen_make_M7_submatrix(
            A_1_2, m1, k2, incRowA,
            A_2_2, m2, k2, incRowA,
            B_2_1, k2, n1, incRowB,
            B_2_2, k2, n2, incRowB);

    /*
    compute C_1_1 by the formula
    C_1_1 = M1 + M4 - M5 + M7
    */
    strassen_calculate_C_1_1(
        M1, m1, n1,
        M4, m2, n1,
        M5, m1, n2,
        M7, m1, n1,
        C_1_1, m1, n1, incRowC);

    /*
    compute C_1_2 by the formula
    C_1_2 = M3 + M5
    */
    strassen_calculate_C_1_2(
        M3, m1, n2,
        M5, m1, n2,
        C_1_2, m1, n2, incRowC);


    /*
    compute C_2_1 by the formula
    C_2_1 = M2 + M4
    */
    strassen_calculate_C_2_1(
        M2, m2, n1,
        M4, m2, n1,
        C_2_1, m2, n1, incRowC);

    /*
    compute C_2_2 by the formula
    C_2_2 = M1 - M2 + M3 + M6
    */
    strassen_calculate_C_2_2(
        M1, m1, n1,
        M2, m2, n1,
        M3, m1, n2,
        M6, m1, n1,
        C_2_2, m2, n2, incRowC);

    // /*
    // remove the working space
    // */
    remove_matrix(M1);
    remove_matrix(M2);
    remove_matrix(M3);
    remove_matrix(M4);
    remove_matrix(M5);
    remove_matrix(M6);
    remove_matrix(M7);
    return;
}

#endif