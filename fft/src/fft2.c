#include "fft_setting.h"
#include <complex.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"


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
					FFT_Type exponent = 2*PI*I*((float)(x*u)/M + (float)(y*v)/N);
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
					FFT_Type exponent = 2*PI*I*((float)(x*(u+su))/M + (float)(y*(v+sv))/N);
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


static void fft_dot_product(unsigned int length,
					 FourierDomain2D* restrict Af,
					 FourierDomain2D* restrict Bf,
					 WriteBackType wbt,
					 FourierDomain2D* restrict Cf){
	if(wbt == FFT_ACCUMULATE){
		for(int i = 0; i < length; i++){
			Cf[i] += Af[i] * Bf[i];
		}
	}
	else{
		for(int i = 0; i < length; i++){
			Cf[i] = Af[i] * Bf[i];
		}
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

static FFT_Type* _padImageTo(unsigned int M,
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

float* _padImageToFloat(unsigned int M,
				  unsigned int N,
				  float* input,
				  unsigned int M_new,
				  unsigned int N_new){
	
	assert(M < M_new);
	assert(N < N_new);
	float* output = (float*)malloc(sizeof(float)*M_new*N_new);
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

static float* copyReverseFloat(unsigned int length,
					float* list){
	float* new_list = (float*)malloc(sizeof(float*)*length);
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
	fft_dot_product(M*N, Af, Bf, FFT_OVERWRITE, temp);
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
	fft_dot_product(M*N, Af, Bf, FFT_OVERWRITE, temp);
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

	float coefficient = 1 / (M * N);
	for(int u = 0; u < M; u++){
		for(int v = 0; v < N; v++){
			FFT_Type value = 0;

			for(int x = 0; x < M; x++){
				for(int y = 0; y < N; y++){
					FFT_Type exponent = -2*PI*I*((float)(x*u)/M + (float)(y*v)/N);
					FFT_Type ce = complex_exponential(exponent);
					FFT_Type input_value = input[x*N+y];
					value += input_value * ce;
				}
			}
			output[u*N+v] = value;
		}
	}
}


void IFFT2s(unsigned int M,
		   unsigned int N,
		   FourierDomain2D* input,
		   const WriteBackType wbt,
		   float* output){

	float coefficient = 1 / (float)(M * N);
	for(int u = 0; u < M; u++){
		for(int v = 0; v < N; v++){
			FFT_Type value = 0;

			for(int x = 0; x < M; x++){
				for(int y = 0; y < N; y++){
					FFT_Type exponent = 2*PI*I*((float)(x*u)/M + (float)(y*v)/N);
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

void _IFFT2s_Shifted(unsigned int M,
		   		   unsigned int N,
		   		   FourierDomain2D* input,
		   		   unsigned int Mo,
		   		   unsigned int No,
		   		   WriteBackType wbt,
		   		   float* output){

	float coefficient = 1 / (float)(M * N);
	unsigned int su = M - Mo;
	unsigned int sv = N - No;
	assert(su >= 0);
	assert(sv >= 0);
	for(int u = 0; u < Mo; u++){
		for(int v = 0; v < No; v++){
			FFT_Type value = 0;

			for(int x = 0; x < M; x++){
				for(int y = 0; y < N; y++){
					FFT_Type exponent = 2*PI*I*((float)(x*(u+su))/M + (float)(y*(v+sv))/N);
					FFT_Type ce = complex_exponential(exponent);
					value += input[x*N+y] * ce;
				}
			}
			if(wbt == FFT_OVERWRITE)
				output[u*No+v] = creal(value) * coefficient;
			else{
				if(wbt == FFT_ACCUMULATE){
					output[u*No+v] += creal(value) * coefficient;
				}
				else{
					assert(0);
				}
			}
		}
	}
}


void constructFTSetFromFloat(float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet, unsigned int Mo, unsigned int No){
	if(Mi == Mo && Ni == No){
		for(int i = 0; i < Di; i++){
			FFT2s(Mi, Ni, &input[i*Mi*Ni], &FTSet[i*Mo*No]);
		}		
	}
	else{
		for(int i = 0; i < Di; i++){
			float* big_image  = _padImageToFloat(Mi, Ni, &input[i*Mi*Ni], Mo, No);
			FFT2s(Mi, Ni, big_image, &FTSet[i*Mo*No]);
			free(big_image);
		}
	}
}

void constructFTSetFromFloatReverse(float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet, unsigned int Mo, unsigned int No){	

	if(Mi == Mo && Ni == No){
		for(int i = 0; i < Di; i++){
			float* r = copyReverseFloat(Mi*Ni, &input[i*Mi*Ni]);
			FFT2s(Mi, Ni, r, &FTSet[i*Mo*No]);
			free(r);
		}
	}
	else{
		for(int i = 0; i < Di; i++){
			float* r = copyReverseFloat(Mi*Ni, &input[i*Mi*Ni]);
			float* big_image  = _padImageToFloat(Mi, Ni, r, Mo, No);
			FFT2s(Mo, No, big_image, &FTSet[i*Mo*No]);
			free(big_image);
			free(r);
		}
	}
}

static void _PairwiseMultiplyInFourierDomain_And_SumUp_(
							FourierDomain2D* A, 
							FourierDomain2D* B,
							unsigned int num_of_pair,
							unsigned int size_of_each_pair,
							FourierDomain2D* C){
	fft_dot_product(size_of_each_pair, A, B, FFT_OVERWRITE, C);
	for(int i = 1; i < num_of_pair; i++){
		fft_dot_product(size_of_each_pair, &A[i*size_of_each_pair], &B[i*size_of_each_pair], FFT_ACCUMULATE, C);
	}
}


void LinearCorrelation3InFourierDomain(
							FourierDomain2D* A, unsigned int Ma, unsigned int Na, unsigned int Da,
							FourierDomain2D* B, unsigned int Mb, unsigned int Nb, unsigned int Db,
							WriteBackType wbt,
							float* C, unsigned int Mc, unsigned int Nc){
	assert(Ma*Na == Mb*Nb);
	assert(Da == Db);
	FourierDomain2D* temp = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*Ma*Na);
	_PairwiseMultiplyInFourierDomain_And_SumUp_(A, B, Da, Ma*Na, temp);
	_IFFT2s_Shifted(Ma, Na, temp, Mc, Nc, wbt, C);
}