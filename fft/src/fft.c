#include "fft_setting.h"
#include <complex.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"
#include <fftw3.h>

#define FFT_FFTW3 FFT
#define IFFT_FFTW3 IFFT

void FFT_Baseline(unsigned int M,
		 FFT_Type* input,
		 FourierDomain2D* output){

	for(int u = 0; u < M; u++){
		FFT_Type value = 0;
		for(int x = 0; x < M; x++){
			FFT_Type exponent = -2*PI*I*((float)(x*u)/M);
			FFT_Type ce = complex_exponential(exponent);
			value += input[x] * ce;
		}
		output[u] = value;
	}

}

void IFFT_Baseline(unsigned int M,
		FourierDomain2D* input,
		FFT_Type* output){
	
	float coefficient = 1.0 / M;	
	for(int u = 0; u < M; u++){
		FFT_Type value = 0;
		for(int x = 0; x < M; x++){
			FFT_Type exponent = 2*PI*I*((float)(x*u)/M);
			FFT_Type ce = complex_exponential(exponent);
			value += input[x] * ce;
		}
		output[u] = value * coefficient;
	}	

}


void FFT_Cooley_Tukey_Worker(unsigned int length,
					  FFT_Type* input,
					  FourierDomain2D* output,
					  unsigned int increments){
	assert(length > 0);
	if(length == 1){
		output[0] = input[0];
		return;
	}
	else{
		unsigned int odd_length = length /2;
		unsigned int even_length = length - even_length;
		FFT_Cooley_Tukey_Worker(even_length, input, output, 2*increments);
		FFT_Cooley_Tukey_Worker(odd_length, &input[increments], &output[length/2], 2*increments);


		for(int k = 0; k < length/2; k++){
			FourierDomain2D temp = output[k];
			output[k] = temp + complex_exponential(-2*PI*I*k/length) * output[k+length/2];
			output[k+length/2] = temp - complex_exponential(-2*PI*I*k/length) * output[k+length/2];
		}
	}
}

void FFT_Cooley_Tukey(unsigned int M,
		 FFT_Type* input,
		 FourierDomain2D* output){

	return FFT_Cooley_Tukey_Worker(M, input, output, 1);
}

void FFT_FFTW3(unsigned int N,
		 FFT_Type* input,
		 FourierDomain2D* output){
		 
		 fftwf_complex *in, *out;
         fftwf_plan p;
         in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
         out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
         for(int i = 0; i < N; i++){
         	in[i] = input[i];
         }
         p = fftwf_plan_dft_1d(N, in, out, fftwf_FORWARD, fftwf_ESTIMATE);
         fftwf_execute(p); /* repeat as needed */
         for(int i = 0; i < N; i++){
         	output[i] = out[i];
         }
         fftwf_destroy_plan(p);
         fftwf_free(in); 
         fftwf_free(out);
}


void IFFT_FFTW3(unsigned int N,
		 FFT_Type* input,
		 FourierDomain2D* output){
		 
		 fftwf_complex *in, *out;
         fftwf_plan p;
         in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
         out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
         for(int i = 0; i < N; i++){
         	in[i] = input[i];
         }
         p = fftwf_plan_dft_1d(N, in, out, fftwf_BACKWARD, fftwf_ESTIMATE);
         fftwf_execute(p); /* repeat as needed */
         for(int i = 0; i < N; i++){
         	output[i] = out[i];
         }
         fftwf_destroy_plan(p);
         fftwf_free(in); 
         fftwf_free(out);
}