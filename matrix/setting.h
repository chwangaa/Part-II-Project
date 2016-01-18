#ifndef MATRIX_SETTING_H
#define MATRIX_SETTING_H

// #define DEBUG
#define GEMM
#ifdef GEMM
	typedef float Dtype;
#else
	typedef int Dtype;
#endif

#define limit_K 512
#define limit_M 128
#define limit_N 128

#ifdef DEBUG
#define debug_assert(...) do{ assert(__VA_ARGS__); } while( false )
#else
#define debug_assert(...) do{ } while ( false )
#endif

/// packed height of A
#define MC  128
/// packed width of A, height of B
#define KC  512
/// packed width of B
#define NC  1024
/// width of micro kernel
#define MR  8
/// height of micro kernel
#define NR  8

#endif
