#ifndef FFT_H
#define FFT_H
#include <complex.h>
#include "fft_setting.h"


void FFT2(unsigned int M,
		  unsigned int N,
		  FFT_Type* input,
		  FourierDomain2D* output);

void IFFT2(unsigned int M,
		   unsigned int N,
		   FourierDomain2D* input,
		   WriteBackType wbt,
		   FFT_Type* output
);

void LinearCorrelation2(FFT_Type* input, unsigned Mi, unsigned Ni,
		   				FFT_Type* kernel, unsigned Mk, unsigned Nk,
		   				WriteBackType wbt,
		   				FFT_Type* output);

void LinearCorrelation3(FFT_Type* input, unsigned Mi, unsigned Ni, unsigned Di,
						FFT_Type* kernel, unsigned Mk, unsigned Nk, unsigned Dk,
		   				WriteBackType wbt,
						FFT_Type* output);

void _IFFT2_Shifted(unsigned int M,
		   		   unsigned int N,
		   		   FourierDomain2D* input,
		   		   unsigned int Mo,
		   		   unsigned int No,
		   		   WriteBackType wbt,
		   		   FFT_Type* output);

void CONV2(FFT_Type* A, unsigned MA, unsigned NA,
		   FFT_Type* B, unsigned MB, unsigned NB,
		   WriteBackType wbt,
		   FFT_Type* C);

void fft_dot_product(unsigned int M,
					 unsigned int N,
					 FourierDomain2D* Af,
					 FourierDomain2D* Bf,
					 FourierDomain2D* cF);

FourierDomain2D* giveFFT2(unsigned int M,
			    unsigned int N,
			    FFT_Type* input);

FFT_Type* giveIFFT2(unsigned int M,
	             unsigned int N,
	             WriteBackType wbt,
	             FourierDomain2D* input);

FFT_Type* padImageTo(unsigned int M,
					  unsigned int N,
					  FFT_Type* input,
					  unsigned int M_new,
					  unsigned int N_new);



#endif