#ifndef FFT_H
#define FFT_H
#include <complex.h>
#include "fft_setting.h"
#include <fftw3.h>


void FFT(unsigned int M,
		 FFT_Type* input,
		 FourierDomain2D* output);

void FFTs(unsigned int M,
		 float* input,
		 FourierDomain2D* output);

void IFFT(unsigned int M,
		FourierDomain2D* input,
		FFT_Type* output);

void FFT2(unsigned int M,
		  unsigned int N,
		  FFT_Type* input,
		  FourierDomain2D* output);

void FFT2s(unsigned int M,
		  unsigned int N,
		  float* input,
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

void _IFFT2s_Shifted(unsigned int M,
		   		   unsigned int N,
		   		   FourierDomain2D* input,
		   		   unsigned int Mo,
		   		   unsigned int No,
		   		   WriteBackType wbt,
		   		   float* output);


void CONV2(FFT_Type* A, unsigned MA, unsigned NA,
		   FFT_Type* B, unsigned MB, unsigned NB,
		   WriteBackType wbt,
		   FFT_Type* C);

FourierDomain2D* giveFFT2(unsigned int M,
			    unsigned int N,
			    FFT_Type* input);

FFT_Type* giveIFFT2(unsigned int M,
	             unsigned int N,
	             WriteBackType wbt,
	             FourierDomain2D* input);

void LinearCorrelation3InFourierDomain(
							FourierDomain2D* A, unsigned int Ma, unsigned int Na, unsigned int Da,
							FourierDomain2D* B, unsigned int Mb, unsigned int Nb, unsigned int Db,
							WriteBackType wbt,
							float* C, unsigned int Mc, unsigned int Nc);

void constructFTSetFromFloat(float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet, unsigned int Mo, unsigned int No);

void constructFTSetFromFloatGivenPlan(fftwf_plan p, float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet, unsigned int Mo, unsigned int No);

void constructFTSetFromFloatReverse(float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet, unsigned int Mo, unsigned int No);


void FFT2s(unsigned int M,
		  unsigned int N,
		  float* input,
		  FourierDomain2D* output);

void IFFT2s(unsigned int M,
		   unsigned int N,
		   FourierDomain2D* input,
		   WriteBackType wbt,
		   float* output
);

#endif