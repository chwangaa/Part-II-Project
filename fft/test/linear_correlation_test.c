#include "fft.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"


/*
	input_image: [[1,2,3],[4,5,6],[7,8,9]]
	kernel: [[1,0],[0,1]]
	output: [[6,8],[12,14]]
*/
void test1(){
	unsigned int M = 3;
	unsigned int N = 3;
	FFT_Type* input = (FFT_Type*)malloc(sizeof(FFT_Type*)*9);
	FFT_Type* kernel = (FFT_Type*)malloc(sizeof(FFT_Type*)*4);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*4);
	assert(input);
	assert(kernel);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = i + 1;
	}
	kernel[0] = 1;
	kernel[1] = 0;
	kernel[2] = 0;
	kernel[3] = 1;

	LinearCorrelation2(input, 3, 3,
		  			   kernel, 2, 2,
		  			   0,
		  			   output);

	FFT_Type correct[4] = {6,8,12,14};
	for(int i = 0; i < 4; i++){
		// print_complex(output[i]);
		assert(isEqual(output[i],correct[i]));
	}

	fprintf(stderr, "the first linear correlation test passes\n");
	free(input);
	free(output);
	free(kernel);
}

/*
	Input: [[1,2,3],[4,5,6],[7,8,9]]
	Kernel: [[1,1],[1,1]]
	Output: [[12,16],[24,28]]
*/
void test2(){
	unsigned int M = 3;
	unsigned int N = 3;
	unsigned int MB = 2;
	unsigned int NB = 2;
	// FFT_Type* A = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* B = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* C = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// assert(C);
	FFT_Type A[9] = {1,2,3,4,5,6,7,8,9};
	FFT_Type B[9] = {1,1,1,1};
	FFT_Type C[4];
	FFT_Type Correct[9] = {12,16,24,28};
	LinearCorrelation2(A, M, N,
		  			   B, MB, NB,
		  			   0,
		  			   C);
	for(int i = 0; i < 4; i++){
		assert(isEqual(C[i], Correct[i]));
		// print_complex(C[i]);
	}

	fprintf(stderr, "the second linear correlation test passes\n");
}

int main(){
	test1();
	test2();
}