#include "fft.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"

/*
	input: [[1,1,1],[1,1,1],[1,1,1]]
	output: [[1,1,1],[1,1,1],[1,1,1]]
*/
void test1(){
	unsigned int M = 3;
	unsigned int N = 3;
	FourierDomain2D* input = (FourierDomain2D*)malloc(sizeof(FourierDomain2D*)*M*N);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			input[i*N+j] = 1.0 + 0.0*I;
		}
	}

	FFT2(M, N,
		input,
		output);

	IFFT2(M, N,
		output,
		0,
		input);

	for(int i = 0; i < M*N; i++){
		assert(isEqual(input[i],1.0+0*I));
		// print_complex(input[i]);
	}

	fprintf(stderr, "the first test passes\n");
	free(input);
	free(output);
}

/*
	input: [[1,1,1],[1,1,1],[1,1,1]]
	output: [[1,1,1],[1,1,1],[1,1,1]]
*/
void test2(){
	unsigned int M = 3;
	unsigned int N = 3;
	FourierDomain2D* input = (FourierDomain2D*)malloc(sizeof(FourierDomain2D*)*M*N);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = i;
	}

	FFT2(M, N,
		input,
		output);

	IFFT2(M, N,
		output,
		0,
		input);

	for(int i = 0; i < M*N; i++){
		assert(isEqual(input[i],i));
		// print_complex(input[i]);
	}

	fprintf(stderr, "the second test passes\n");
	free(input);
	free(output);
}


/*
	input: [[1,2,3],[4,5,6],[7,8,9,],[10,11,12]]
	output:[[1,2,3],[4,5,6],[7,8,9,],[10,11,12]]
*/
void test3(){
	unsigned int M = 10;
	unsigned int N = 20;
	FourierDomain2D* input = (FourierDomain2D*)malloc(sizeof(FourierDomain2D*)*M*N);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = 0.1;
	}

	FFT2(M, N,
		input,
		output);

	IFFT2(M, N,
		output,
		0,
		input);

	for(int i = 0; i < M*N; i++){
		assert(isEqual(input[i],0.1));
		// print_complex(input[i]);
	}

	fprintf(stderr, "the third test passes\n");
	free(input);
	free(output);
}

int main(){
	test1();
	test2();
	test3();
}