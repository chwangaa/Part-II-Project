#include "fft_setting.h"
#include <complex.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"
#ifndef PI
	#define PI 3.14159265358979323846
#endif

FFT_Type complex_exponential(FFT_Type x){
	return cexpf(x);
}


void FFT2(unsigned int M,
		  unsigned int N,
		  FFT_Type* input,
		  FourierDomain2D* output){

	double coefficient = 1 / (M * N);
	for(int u = 0; u < M; u++){
		for(int v = 0; v < N; v++){
			FFT_Type value = 0;

			for(int x = 0; x < M; x++){
				for(int y = 0; y < N; y++){
					FFT_Type exponent = -2*PI*I*((float)(x*u)/M + (float)(y*v)/N);
					FFT_Type ce = complex_exponential(exponent);
					value += input[x*N+y] * ce;
				}
			}
			output[u*N+v] = value;
		}
	}
}


void IFFT2(unsigned int M,
		   unsigned int N,
		   FourierDomain2D* input,
		   WriteBackType wbt,
		   FFT_Type* output){

	double coefficient = 1 / (float)(M * N);
	for(int u = 0; u < M; u++){
		for(int v = 0; v < N; v++){
			FFT_Type value = 0;

			for(int x = 0; x < M; x++){
				for(int y = 0; y < N; y++){
					FFT_Type exponent = 2*PI*I*((float)(x*u)/N + (float)(y*v)/M);
					FFT_Type ce = complex_exponential(exponent);
					value += input[x*N+y] * ce;
				}
			}
			if(wbt == FFT_OVERWRITE)
				output[u*N+v] = value * coefficient;
			else{
				if(wbt == FFT_ACCUMULATE){
					output[u*N+v] += value * coefficient;
				}
				else{
					assert(0);
				}
			}
		}
	}
}


void _IFFT2_Shifted(unsigned int M,
		   		   unsigned int N,
		   		   FourierDomain2D* input,
		   		   unsigned int Mo,
		   		   unsigned int No,
		   		   WriteBackType wbt,
		   		   FFT_Type* output){

	double coefficient = 1 / (float)(M * N);
	unsigned int su = M - Mo;
	unsigned int sv = N - No;
	assert(su >= 0);
	assert(sv >= 0);
	for(int u = 0; u < Mo; u++){
		for(int v = 0; v < No; v++){
			FFT_Type value = 0;

			for(int x = 0; x < M; x++){
				for(int y = 0; y < N; y++){
					FFT_Type exponent = 2*PI*I*((float)(x*(u+su))/N + (float)(y*(v+sv))/M);
					FFT_Type ce = complex_exponential(exponent);
					value += input[x*N+y] * ce;
				}
			}
			if(wbt == FFT_OVERWRITE)
				output[u*No+v] = value * coefficient;
			else{
				if(wbt == FFT_ACCUMULATE){
					output[u*No+v] += value * coefficient;
				}
				else{
					assert(0);
				}
			}
		}
	}
}


void fft_dot_product(unsigned int M,
					 unsigned int N,
					 FourierDomain2D* Af,
					 FourierDomain2D* Bf,
					 FourierDomain2D* Cf){
	for(int i = 0; i < M*N; i++){
		Cf[i] = Af[i] * Bf[i];
	}
}

FourierDomain2D* giveFFT2(unsigned int M,
			    		  unsigned int N,
			    		  FFT_Type* input){
	
	FourierDomain2D* output = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*M*N);
	assert(output);
	FFT2(M, N, input, output);
	return output;
}

FFT_Type* giveIFFT2(unsigned int M,
	             unsigned int N,
	             WriteBackType wbt,
	             FourierDomain2D* input){
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type)*M*N);
	assert(output);
	IFFT2(M, N, input, wbt, output);
	return output;	
}

FFT_Type* _padImageTo(unsigned int M,
				  unsigned int N,
				  FFT_Type* input,
				  unsigned int M_new,
				  unsigned int N_new){
	
	assert(M < M_new);
	assert(N < N_new);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type)*M_new*N_new);
	assert(output);
	for(int i = 0; i < M; i++){
		memcpy(&output[i*N_new], &input[i*N], sizeof(FFT_Type)*N);
		for(int j = N; j < N_new; j++){
			output[i*N_new + j] = 0;
		}
	}
	for(int i = M; i < M_new; i++){
		for(int j = 0; j < N_new; j++){
			output[i*N_new+j] = 0;
		}
	}
	return output;
}

static void reverse(unsigned int length,
					FFT_Type* list){
	for(int i = 0; i < length/2; i++){
		FFT_Type temp = list[i];
		list[i] = list[length-1-i];
		list[length-1-i] = temp;
	}
}

static FFT_Type* copyReverse(unsigned int length,
					FFT_Type* list){
	FFT_Type* new_list = (FFT_Type*)malloc(sizeof(FFT_Type*)*length);
	assert(new_list);
	for(int i = 0; i < length; i++){
		new_list[i] = list[length-1-i];
	}
	return new_list;
}

static void _CONV2_full(unsigned int M,
		   unsigned int N,
		   FFT_Type* A,
		   FFT_Type* B,
		   WriteBackType wbt,
		   FFT_Type* C){
	FourierDomain2D* Af = giveFFT2(M, N, A);
	FourierDomain2D* Bf = giveFFT2(M, N, B);
	FourierDomain2D* temp = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*M*N);
	fft_dot_product(M, N, Af, Bf, temp);
	IFFT2(M, N, temp, wbt, C);
	free(Af);
	free(Bf);
	free(temp);
}

static void _CONV2_valid(unsigned int M,
		   unsigned int N,
		   FFT_Type* A,
		   unsigned int Mb,
		   unsigned int Nb,
		   FFT_Type* B,
		   WriteBackType wbt,
		   FFT_Type* C){
	assert(Mb <= M);
	assert(Nb <= N);
	unsigned int Mc = M - Mb + 1;
	unsigned int Nc = N - Nb + 1;
	FourierDomain2D* Af = giveFFT2(M, N, A);
	FourierDomain2D* Bf = giveFFT2(M, N, B);
	FourierDomain2D* temp = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*M*N);
	fft_dot_product(M, N, Af, Bf, temp);
	_IFFT2_Shifted(M, N, temp, Mc, Nc, wbt, C);
	free(Af);
	free(Bf);
	free(temp);
}

void CONV2(FFT_Type* A, unsigned MA, unsigned NA,
		   FFT_Type* B, unsigned MB, unsigned NB,
		   WriteBackType wbt,
		   FFT_Type* C){
	if(MA == MB && NA == NB){
		_CONV2_full(MA, NA, A, B, wbt, C);
	}
	else{
		assert(MA >= MB);
		assert(NA >= NB);
		FFT_Type* newB = _padImageTo(MB, NB, B, MA, NA);
		_CONV2_valid(MA, NA, A, MB, NB, newB, wbt, C);
	}
}

void LinearCorrelation2(FFT_Type* input, unsigned Mi, unsigned Ni,
		   FFT_Type* kernel, unsigned Mk, unsigned Nk,
		   WriteBackType wbt,
		   FFT_Type* output){
	FFT_Type* kernel_reversed = copyReverse(Mk*Nk, kernel);
	CONV2(input, Mi, Ni,
		  kernel_reversed, Mk, Nk,
		  wbt,
		  output);
	free(kernel_reversed);
}

void LinearCorrelation3(FFT_Type* input, unsigned Mi, unsigned Ni, unsigned Di,
						FFT_Type* kernel, unsigned Mk, unsigned Nk, unsigned Dk,
						WriteBackType wbt,
						FFT_Type* output){
	assert(Di == Dk);
	// FFT_Type* new_kernels = copyReverse3D(Mk*Nk, kernel);	
	if(wbt == FFT_ACCUMULATE){
		for(int i = 0; i < Di; i++){
			LinearCorrelation2(&input[i*Mi*Ni], Mi, Ni,
							   &kernel[i*Mk*Nk], Mk, Nk,
							   wbt,
							   output);
		}
	}
	else{
		assert(wbt == FFT_OVERWRITE);
		LinearCorrelation2(input, Mi, Ni,
						   kernel, Mk, Nk,
						   FFT_OVERWRITE,
						   output);
		for(int i = 1; i < Di; i++){
			LinearCorrelation2(&input[i*Mi*Ni], Mi, Ni,
							   &kernel[i*Mk*Nk], Mk, Nk,
							   FFT_ACCUMULATE,
							   output);
		}
	}
}

void FFT2s(unsigned int M,
		   unsigned int N,
		   float* input,
		   FourierDomain2D* output){

	double coefficient = 1 / (M * N);
	for(int u = 0; u < M; u++){
		for(int v = 0; v < N; v++){
			FFT_Type value = 0;

			for(int x = 0; x < M; x++){
				for(int y = 0; y < N; y++){
					FFT_Type exponent = -2*PI*I*((float)(x*u)/M + (float)(y*v)/N);
					FFT_Type ce = complex_exponential(exponent);
					value += input[x*N+y] * ce;
				}
			}
			output[u*N+v] = value;
		}
	}
}


void IFFT2s(unsigned int M,
		   unsigned int N,
		   FourierDomain2D* input,
		   WriteBackType wbt,
		   float* output){

	float coefficient = 1 / (float)(M * N);
	for(int u = 0; u < M; u++){
		for(int v = 0; v < N; v++){
			FFT_Type value = 0;

			for(int x = 0; x < M; x++){
				for(int y = 0; y < N; y++){
					FFT_Type exponent = 2*PI*I*((float)(x*u)/N + (float)(y*v)/M);
					FFT_Type ce = complex_exponential(exponent);
					value += input[x*N+y] * ce;
				}
			}
			if(wbt == FFT_OVERWRITE)
				output[u*N+v] = creal(value) * coefficient;
			else{
				if(wbt == FFT_ACCUMULATE){
					output[u*N+v] += creal(value) * coefficient;
				}
				else{
					assert(0);
				}
			}
		}
	}
}


void constructFTSetFromFloat(float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet){
	for(int i = 0; i < Di; i++){
		FFT2s(Mi, Ni, input[i*Mi*Ni], FTSet[i*Mi*Ni]);
	}
}