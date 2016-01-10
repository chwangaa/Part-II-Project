#ifndef MATRIX_SETTING_H
#define MATRIX_SETTING_H

// #define DEBUG
#define GEMM
#ifdef GEMM
	typedef float Dtype;
#else
	typedef int Dtype;
#endif

#define limit_X 256

#ifdef DEBUG
#define debug_assert(...) do{ assert(__VA_ARGS__); } while( false )
#else
#define debug_assert(...) do{ } while ( false )
#endif


#endif
