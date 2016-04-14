#ifndef FFT_UTIL_H
#define FFT_UTIL_H
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#ifndef PI
	#define PI 3.14159265358979323846
#endif

static inline FFT_Type complex_exponential(FFT_Type x){
	return cexpf(x);
}

static inline void print_complex(FFT_Type x){
	fprintf(stderr, "%.4f+%.4fi ", creal(x), cimag(x));
}

static inline bool isEqual(FFT_Type x, FFT_Type y){
	double real_diff = creal(x) - creal(y);
	double imag_diff = cimag(x) - cimag(y); 
	bool real_part_equal = (real_diff < 0.01 && real_diff > -0.01);
	bool imaginary_part_equal = (imag_diff < 0.01 && imag_diff > -0.01);
	return real_part_equal && imaginary_part_equal;
}

static inline bool isEqual_flexible(FFT_Type x, FFT_Type y){
	double real_diff = creal(x) - creal(y);
	double imag_diff = cimag(x) - cimag(y); 
	bool real_part_equal = (real_diff < 0.1 && real_diff > -0.1);
	bool imaginary_part_equal = (imag_diff < 0.1 && imag_diff > -0.1);
	return real_part_equal && imaginary_part_equal;
}

inline void padOneChannel(float* start, float* new, int pad, int X, int Y){
	assert(start);
	assert(pad >=0 );
	int newX = X + 2*pad;
	int newY = Y + 2*pad;
	// the first few rows will all be 0s
	for(int i = 0; i < pad; i++){
		for(int j = 0; j < newY; j++){
			new[i*newY+j] = 0;
		}
	}
	// then last few rows will all be 0s
	for(int i = X+pad; i < newX; i++){
		for(int j = 0; j < newY; j++){
			new[i*newY+j] = 0;
		}
	}
	// finnally deal with the middle rows
	for(int i = pad; i < X+pad; i++){
		for(int j = 0; j < pad; j++){
			new[i*newY+j] = 0;
		}
		for(int j = pad; j < Y+pad; j++){
			new[i*newY+j] = start[(i-pad)*Y+(j-pad)];
		}
		for(int j = Y+pad; j < newY; j++){
			new[i*newY+j] = 0;
		}
	}
}

inline float* padImage(float* image, int pad, int X, int Y, int Z){
  assert(pad >= 0);
  if(pad == 0){
    return image;
  }
  else{
    int newX = X+2*pad;
    int newY = Y+2*pad;
    float* padded = (float*)malloc(sizeof(float)*(newX)*(newY)*Z);
    assert(padded);
    for(int c = 0; c < Z; c++){
      padOneChannel(&image[X*Y*c], &padded[newX*newY*c], pad, X, Y);
    }
    return padded;
  }
}

#endif