#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

#include "matrix.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>


Dtype* create_matrix(int M, int N, int incRow, Dtype default_value);

#ifdef DEBUG
#define debug_assert(...) do{ assert(__VA_ARGS__); } while( false )
#else
#define debug_assert(...) do{ } while ( false )
#endif

static inline uint64_t timestamp_us() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return 1000000L * tv.tv_sec + tv.tv_usec;
}

#define debug_print(fmt, ...) \
            do { if (DEBUG)fprintf(stderr, fmt, __VA_ARGS__); } while (0)

#define function_summary(func, ...) \
            do { \
              if(DEBUG){ \
              uint64_t start_time = timestamp_us(); \
              func(__VA_ARGS__); \
              uint64_t end_time = timestamp_us(); \
              double dt = (double)(end_time-start_time) / 1000.0; \
              debug_print(#func " takes %lf ms to complete \n", dt); \
              } \
              else{ \
              func(__VA_ARGS__); \
              } \
            } while(0)


Dtype* make_matrix(const unsigned int M, const unsigned int N);
void matrix_addition(
    const unsigned int M,
    const unsigned int N,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC);
void matrix_subtraction(
    const unsigned int M,
    const unsigned int N,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC);
Dtype* pad_matrix(Dtype* old_matrix, const unsigned int old_M, const unsigned int old_N,
                  const unsigned int old_incRow,
                  const unsigned int new_M,
                  const unsigned int new_N);
void remove_matrix(Dtype* old_matrix);

void print_matrix(Dtype* matrix, int M, int N, int incRow);

void matrix_copyTo(
    Dtype* const from_matrix, int M, int N, int incRowFrom,
    Dtype* to_matrix, int M_to, int N_to, int incRowTo);

Dtype* matrix_copy(
    Dtype* const from_matrix, int M, int N, int incRowFrom);

void matrix_partial_addition(Dtype* result, int rM, int rN, int rincRow,
                             const Dtype* adder,  int aM, int aN, int aincRow);

void matrix_partial_subtraction(Dtype* result, int rM, int rN, int rincRow,
                             const Dtype* adder,  int aM, int aN, int aincRow);

Dtype* addDifferentSizedMatrix(
    Dtype* const Larger, int lm, int ln, int incRowL,
    Dtype* const Smaller, int sm, int sn, int incRowS);

Dtype* subtractDifferentSizedMatrix(
    Dtype* const Larger, int lm, int ln, int incRowL,
    Dtype* Smaller, int sm, int sn, int incRowS);

void MakePackedA(Dtype* A, int M, int K, int incRowA,
            int M_target, int K_target, 
            Dtype* newA);

void MakePackedB(Dtype* B, int K, int N, int incRowB,
            int K_target, int N_target, 
            Dtype* newB);

#endif