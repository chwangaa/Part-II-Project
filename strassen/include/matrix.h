#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

typedef float Dtype;

/*
the options are:
    cblas_mm
    blis_mm
    strassen_mm
    SimpleMatrixMultiplication
*/


#define matrix_multiplication strassen_mm
#define matrix_multiplication_base_case SimpleMatrixMultiplication
#define packed_strassen_base_matrix_multiplication packed_mm
#define blas_mm blis_mm

#define DEBUG 0
#include "matrix_util.h"

/*
In this file, a matrix multiplication need to be defined, which should have the following signature

void matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC);
*/



#endif