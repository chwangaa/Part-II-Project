#ifndef STRASSEN_MATRIX_MULTIPLICATION_H
#define STRASSEN_MATRIX_MULTIPLICATION_H

#include "setting.h"

void strassen_matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC);

#endif