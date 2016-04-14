#include "fft.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"
#include <fftw3.h>

void test_fft_from_float(){
	unsigned int M = 10;
	float* input = (float*)malloc(sizeof(float*)*M);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	FFT_Type correct[6] = {45, -5+15.3884177*I, -5+6.88190960*I,
			 -5+3.63271264*I, -5+1.62459848*I, -5};
	assert(input);
	assert(output);
	for(int i = 0; i < M; i++){
		input[i] = i;
		output[i] = 0;
	}

	FFTs(M, input, output);

	for(int i = 0; i < 6; i++){
		// print_complex(output[i]);
		assert(isEqual(output[i], correct[i]));
		assert(isEqual(input[i], i));
	}

	fprintf(stderr, "the FFTs test passes\n");
	free(input);
	free(output);
}

void test_padding(){
	unsigned int M = 3;
	float* input = (float*)malloc(sizeof(float*)*M*M*1);
	assert(input);
	for(int i = 0; i < M*M; i++){
		input[i] = i + 1;
	}
	float correct[5*5] = {0,0,0,0,0,
						  0,1,2,3,0,
						  0,4,5,6,0,
						  0,7,8,9,0,
						  0,0,0,0,0};

	float* output = padImage(input, 1, M, M, 1);
	for(int i = 0; i < 5; i++){
		for(int j = 0; j < 5; j++){
			assert(output[i*5+j] == correct[i*5+j]);
		}
	}

	fprintf(stderr, "the padding test passes\n");
	free(input);
	free(output);
}


/*
	This test test whether the fft works properly
	input: [[1,2,3],[4,5,6],[7,8,9]]
	output: [[ 45.0+0.j        ,  -4.5+2.59807621j,  -4.5-2.59807621j],
       		 [-13.5+7.79422863j,   0.0+0.j        ,   0.0+0.j        ],
       		 [-13.5-7.79422863j,   0.0+0.j        ,   0.0+0.j        ]
       		])

*/
void test_fft2s(){
	unsigned int M = 3;
	unsigned int N = 3;
	float* input = (float*)malloc(sizeof(float*)*M*N);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M*N; i++){
		input[i] = i + 1;
	}

	FFT2s(M, N,
		input,
		output);
	assert(isEqual(output[0],45.0 + 0*I));
	assert(isEqual(output[1],-4.5+2.5980762*I));
	assert(isEqual(output[2],-13.5+7.7942286*I));
	assert(isEqual(output[3],0+0*I));
	assert(isEqual(output[4],-13.5-7.7942286*I));
	assert(isEqual(output[5],0+0*I));

	fprintf(stderr, "the second test passes\n");
	free(input);
	free(output);
}

void test_ifft2s(){
	unsigned int M = 3;
	unsigned int N = 3;
	float* input = (float*)malloc(sizeof(float*)*M*N);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M*N; i++){
		input[i] = i + 1;
	}

	FFT2s(M, N,
		input,
		output);

	IFFT2s(M, N,
		output,
		0,
		input);

	for(int i = 0; i < M*N; i++){
		assert(isEqual(input[i]/M/N, i+1));
	}

	fprintf(stderr, "the ifft2s test passes\n");
	free(input);
	free(output);
}

/*
	input: [[1,2,3],[4,5,6],[7,8,9]]
	output: [[5,6],[8,9]]
*/
void shifted_ifft2s_test(){
	unsigned int M = 3;
	unsigned int N = 3;
	float* input = (float*)malloc(sizeof(float*)*M*N);
	float* output = (float*)malloc(sizeof(float*)*M*N);
	FourierDomain2D* temp = (FourierDomain2D*)malloc(sizeof(FourierDomain2D*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = i + 1;
		output[i] = 0;
	}

	FFT2s(M, N,
		input,
		temp);

	_IFFT2s_Shifted(M, N,
		temp,
		M-1, N-1,
		0,
		output);

	FFT_Type correct[4] = {5,6,8,9};
	for(int i = 0; i < 4; i++){
		assert(isEqual(output[i],correct[i]));
	}

	fprintf(stderr, "the shifted ifft2 test passes\n");
	free(input);
	free(output);
}


/*
	input_image: [[1,2,3],[4,5,6],[7,8,9]]
	kernel: [[1,0],[0,1]]
	output: [[6,8],[12,14]]
*/
void linear_correlation_test(){
	unsigned int M = 3;
	unsigned int N = 3;
	float* input = (float*)malloc(sizeof(float*)*9);
	float* kernel = (float*)malloc(sizeof(float*)*4);
	float* output = (float*)malloc(sizeof(float*)*9);
	assert(input);
	assert(kernel);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = i + 1;
		output[i] = 0;
	}
	kernel[0] = 1;
	kernel[1] = 0;
	kernel[2] = 0;
	kernel[3] = 1;

	FourierDomain2D* Af = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*3*3*1);
	FourierDomain2D* Bf = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*3*3*1);
	constructFTSetFromFloat(input, 3, 3, 1, 
							Af, 3, 3);

	constructFTSetFromFloatReverse(kernel, 2, 2, 1,
								   Bf, 3, 3);
	LinearCorrelation3InFourierDomain(Af, 3, 3, 1,
									  Bf, 3, 3, 1,
									  FFT_ACCUMULATE,
									  output, 2, 2);

	FFT_Type correct[4] = {6,8,12,14};
	for(int i = 0; i < 4; i++){
		assert(isEqual(output[i],correct[i]));
	}

	fprintf(stderr, "the 3d linear correlation test passes\n");
	free(input);
	free(output);
	free(kernel);
}

/*
	input_image: [[1,2,3],[4,5,6],[7,8,9]]
	kernel: [[1,0],[0,1]]
	output: [[6,8],[12,14]]
*/
void linear_correlation_test_with_padding(){
	unsigned int M = 4;
	unsigned int N = 4;
	float* input = (float*)malloc(sizeof(float*)*M*N);
	float* kernel = (float*)malloc(sizeof(float*)*2*2);
	float* output = (float*)malloc(sizeof(float*)*5*5);
	assert(input);
	assert(kernel);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = 1;
		output[i] = 0;
	}
	kernel[0] = 1;
	kernel[1] = 1;
	kernel[2] = 1;
	kernel[3] = 1;

	FourierDomain2D* Af = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*6*6*1);
	FourierDomain2D* Bf = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*6*6*1);
	float* padded_image = padImage(input, 1, M, N, 1);
	constructFTSetFromFloat(padded_image, 6, 6, 1, 
							Af, 6, 6);
	constructFTSetFromFloatReverse(kernel, 2, 2, 1,
								   Bf, 6, 6);
	LinearCorrelation3InFourierDomain(Af, 6, 6, 1,
									  Bf, 6, 6, 1,
									  FFT_ACCUMULATE,
									  output, 5, 5);

	FFT_Type correct[25] = {1,2,2,2,1,
							2,4,4,4,2,
							2,4,4,4,2,
							2,4,4,4,2,
							1,2,2,2,1};

	for(int i = 0; i < 25; i++){
		assert(isEqual(output[i],correct[i]));
	}

	fprintf(stderr, "the padded linear correlation test passes\n");
	free(input);
	free(output);
	free(kernel);
	free(padded_image);
}

/*
	input_image: [[1,2,3],[4,5,6],[7,8,9]]
	kernel: [[1,0],[0,1]]
	output: [[6,8],[12,14]]
*/
void linear_correlation_test2(){
	unsigned int M = 3;
	unsigned int N = 3;
	float* input = (float*)malloc(sizeof(float*)*9);
	float* kernel = (float*)malloc(sizeof(float*)*4);
	float* output = (float*)malloc(sizeof(float*)*9);
	assert(input);
	assert(kernel);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = i + 1;
		output[i] = 0;
	}
	kernel[0] = 1;
	kernel[1] = 0;
	kernel[2] = 0;
	kernel[3] = 1;

	FourierDomain2D* Af = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*3*3*1);
	FourierDomain2D* Bf = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*3*3*1);
	
    fftwf_plan p = fftwf_plan_dft_r2c_2d(M, N, input, Af,
                                    	 FFTW_ESTIMATE);

	constructFTSetFromFloatGivenPlan(p, input, 3, 3, 1, 
							Af, 3, 3);

	constructFTSetFromFloatReverse(kernel, 2, 2, 1,
								   Bf, 3, 3);
	LinearCorrelation3InFourierDomain(Af, 3, 3, 1,
									  Bf, 3, 3, 1,
									  FFT_ACCUMULATE,
									  output, 2, 2);

	FFT_Type correct[4] = {6,8,12,14};
	for(int i = 0; i < 4; i++){
		assert(isEqual(output[i],correct[i]));
	}

	fprintf(stderr, "the 2nd 3d linear correlation test passes\n");
	free(input);
	free(output);
	free(kernel);
}


void spatial_linear_correlation_test(){

	float A[48] = {1,0,0,0,
				   0,1,0,0,
				   0,0,1,0,
				   0,0,0,1,
				   1,2,3,4,
				   5,6,7,8,
				   9,10,11,12,
				   13,14,15,16,
				   1,1,1,1,
				   1,1,1,1,
				   1,1,1,1,
				   1,1,1,1};
	float B[12] = {0,2,
	               3,0,
	               1,0,
	               0,1,
	               1,1,
	               1,1};
	float C[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

	float Correct[9] = {11, 16, 15,
	                    21, 21, 26,
	                    27, 31, 31};
	
	FourierDomain2D* Af = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*4*4*3);
	FourierDomain2D* Bf = (FourierDomain2D*)malloc(sizeof(FourierDomain2D)*4*4*3);

    fftwf_plan p = fftwf_plan_dft_r2c_2d(4, 4, A, Af,
                                    	 FFTW_ESTIMATE);

	constructFTSetFromFloatGivenPlan(p, A, 4, 4, 3, 
							Af, 4, 4);

	constructFTSetFromFloatReverse(B, 2, 2, 3,
								   Bf, 4, 4);
	LinearCorrelation3InFourierDomain(Af, 4, 4, 3,
									  Bf, 4, 4, 3,
									  FFT_ACCUMULATE,
									  C, 3, 3);
	for(int i = 0; i < 9; i++){
		// print_complex(C[i]);
		assert(C[i]-Correct[i] < 0.0001 && C[i] - Correct[i] > -0.0001);
	}

	fprintf(stderr, "the 3d spatial linear correlation test passes\n");
}

void spatial_linear_correlation_with_padding_test(){

	float A[48] = {1,0,0,0,
				   0,1,0,0,
				   0,0,1,0,
				   0,0,0,1,
				   1,2,3,4,
				   5,6,7,8,
				   9,10,11,12,
				   13,14,15,16,
				   1,1,1,1,
				   1,1,1,1,
				   1,1,1,1,
				   1,1,1,1};
	float B[12] = {0,2,
	               3,0,
	               1,0,
	               0,1,
	               1,1,
	               1,1};
	float Correct[49] = {
				   0,0,0,0,0,0,0,
				   0,2,7,5,6,1,0,
 				   0,9,11,16,15,6,0,
				   0,11,21,21,26,10,0,
				   0,15,27,31,31,17,0,
				   0,1,15,16,19,17,0,
				   0,0,0,0,0,0,0};


	float C[49] = {0,0,0,0,0,0,0,
				   0,0,0,0,0,0,0,
				   0,0,0,0,0,0,0,
				   0,0,0,0,0,0,0,
				   0,0,0,0,0,0,0,
				   0,0,0,0,0,0,0,
				   0,0,0,0,0,0,0};
	
    FourierDomain2D* imageFFT = (FourierDomain2D*)malloc(sizeof(FourierDomain2D*)*8*8*3);
    float* image_padded = padImage(A, 2, 4, 4, 3);
    constructFTSetFromFloat(image_padded, 8, 8, 3, imageFFT, 8, 8);
    FourierDomain2D* kernelFFT = (FourierDomain2D*)malloc(sizeof(FourierDomain2D*)*8*8*3);
	constructFTSetFromFloatReverse(B, 2, 2, 3,
								   kernelFFT, 8, 8);
	LinearCorrelation3InFourierDomain(imageFFT, 8, 8, 3,
									  kernelFFT, 8, 8, 3,
									  FFT_ACCUMULATE,
									  C, 7, 7);
	for(int i = 0; i < 49; i++){
		assert(C[i]-Correct[i] < 0.0001 && C[i] - Correct[i] > -0.0001);
	}

	fprintf(stderr, "the padded 3d spatial linear correlation test passes\n");
}


int main(){
	fprintf(stderr, "start testing fftw module \n\n");
	test_padding();	
	test_fft_from_float();
	test_fft2s();
	test_ifft2s();
	shifted_ifft2s_test();
	linear_correlation_test();
	linear_correlation_test_with_padding();
	linear_correlation_test2();
	spatial_linear_correlation_test();
spatial_linear_correlation_with_padding_test();
}
