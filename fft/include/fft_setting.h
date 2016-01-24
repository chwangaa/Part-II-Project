#ifndef FFT_SETTING_H
#define FFT_SETTING_H
#include <complex.h>
typedef float complex FFT_Type;
typedef FFT_Type FourierDomain2D;

typedef enum{
	FFT_ACCUMULATE = 1,
	FFT_OVERWRITE = 0
} WriteBackType;
#endif