#ifndef FFT_UTIL_H
#define FFT_UTIL_H
#include <complex.h>
#include <stdbool.h>
#include <math.h>

static inline void print_complex(FFT_Type x){
	fprintf(stderr, "%.2f+%.2fi ", creal(x), cimag(x));
}

static inline bool isEqual(FFT_Type x, FFT_Type y){
	double real_diff = creal(x) - creal(y);
	double imag_diff = cimag(x) - cimag(y); 
	bool real_part_equal = (real_diff < 0.1 && real_diff > -0.1);
	bool imaginary_part_equal = (imag_diff < 0.1 && imag_diff > -0.1);
	return real_part_equal && imaginary_part_equal;
}

#endif