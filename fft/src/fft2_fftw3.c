#include "fft_setting.h"
#include <complex.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"
#include <fftw3.h>

void FFTs(unsigned int M,
		   float* input,
		   FourierDomain2D* output){


     fftwf_plan p = fftwf_plan_dft_r2c_1d(M, input, output,
                                    	FFTW_PRESERVE_INPUT);

     fftwf_execute(p); /* repeat as needed */
     fftwf_destroy_plan(p);
}


void FFT2s(unsigned int M,
		   unsigned int N,
		   float* input,
		   FourierDomain2D* output){

    fftwf_plan p = fftwf_plan_dft_r2c_2d(M, N, input, output,
                                    	 FFTW_ESTIMATE);    

    fftwf_execute(p);
    fftwf_destroy_plan(p);
}

void IFFT2s(unsigned int M,
		   unsigned int N,
		   FourierDomain2D* input,
		   WriteBackType wbt,
		   float* output){

    fftwf_plan p = fftwf_plan_dft_c2r_2d(M, N, input, output,
                                    	 FFTW_ESTIMATE);

    fftwf_execute(p);
    fftwf_destroy_plan(p);
}

void _IFFT2s_Shifted(unsigned int M,
		   		   unsigned int N,
		   		   FourierDomain2D* input,
		   		   unsigned int Mo,
		   		   unsigned int No,
		   		   WriteBackType wbt,
		   		   float* output){

    float* out = (float*) fftwf_malloc(sizeof(float) * M * N);     
    fftwf_plan p = fftwf_plan_dft_c2r_2d(M, N, input, out,
                                    	 FFTW_ESTIMATE);

    fftwf_execute(p);
    int factor = M * N;
    for(int i = 0; i < Mo; i++){
    	for(int j = 0; j < No; j++){
    		output[i*No+j] += out[(i+(M-Mo))*N+j+(N-No)] / factor;
    	}
    }
    fftwf_free(out);
    fftwf_destroy_plan(p);

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

void constructFTSetFromFloat(float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet, unsigned int Mo, unsigned int No){

    fftwf_plan p = fftwf_plan_dft_r2c_2d(Mi, Ni, input, FTSet,
                                    	 FFTW_ESTIMATE);
	if(Mi == Mo && Ni == No){
		for(int i = 0; i < Di; i++){
    		fftwf_execute_dft_r2c(p, &input[i*Mi*Ni], &FTSet[i*Mo*No]);
		}		
	}

	else{
		for(int i = 0; i < Di; i++){
			float* big_image  = _padImageToFloat(Mi, Ni, &input[i*Mi*Ni], Mo, No);
			FFT2s(Mi, Ni, big_image, &FTSet[i*Mo*No]);
			free(big_image);
		}
	}
    fftwf_destroy_plan(p);
}

void constructFTSetFromFloatGivenPlan(fftwf_plan p, float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet, unsigned int Mo, unsigned int No){

	if(Mi == Mo && Ni == No){
		for(int i = 0; i < Di; i++){
    		fftwf_execute_dft_r2c(p, &input[i*Mi*Ni], &FTSet[i*Mo*No]);
		}		
	}

	else{
		for(int i = 0; i < Di; i++){
			float* big_image  = _padImageToFloat(Mi, Ni, &input[i*Mi*Ni], Mo, No);
			FFT2s(Mi, Ni, big_image, &FTSet[i*Mo*No]);
			free(big_image);
		}
	}
    fftwf_destroy_plan(p);
}

void constructFTSetFromFloatReverse(float* input, unsigned int Mi, unsigned int Ni, unsigned int Di,
					FourierDomain2D* FTSet, unsigned int Mo, unsigned int No){	


	if(Mi == Mo && Ni == No){
    	fftwf_plan p = fftwf_plan_dft_r2c_2d(Mi, Ni, input, FTSet,
	                                    	 FFTW_ESTIMATE);
		for(int i = 0; i < Di; i++){
			float* r = copyReverseFloat(Mi*Ni, &input[i*Mi*Ni]);
    		fftwf_execute_dft_r2c(p, r, &FTSet[i*Mo*No]);			
			free(r);
		}
    	fftwf_destroy_plan(p);
	}
	else{
	    fftwf_plan p = fftwf_plan_dft_r2c_2d(Mo, No, input, FTSet,
	                                    	 FFTW_ESTIMATE);
		for(int i = 0; i < Di; i++){
			float* r = copyReverseFloat(Mi*Ni, &input[i*Mi*Ni]);
			float* big_image  = _padImageToFloat(Mi, Ni, r, Mo, No);
    		fftwf_execute_dft_r2c(p, big_image, &FTSet[i*Mo*No]);
			free(big_image);
			free(r);
		}
		fftwf_destroy_plan(p);
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
	free(temp);
}