#include "fft.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"


/*
	input: [[1,2,3],[4,5,6],[7,8,9]]
	output: [[5,6],[8,9]]
*/
void test1(){
	unsigned int M = 3;
	unsigned int N = 3;
	FourierDomain2D* input = (FourierDomain2D*)malloc(sizeof(FourierDomain2D*)*M*N);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = i + 1;
		output[i] = 0;
	}

	FFT2(M, N,
		input,
		output);

	_IFFT2_Shifted(M, N,
		output,
		M-1, N-1,
		0,
		input);

	FFT_Type correct[4] = {5,6,8,9};
	for(int i = 0; i < 4; i++){
		assert(isEqual(input[i],correct[i]));
	}

	fprintf(stderr, "the first test of shifted ifft2 passes\n");
	free(input);
	free(output);
}

int main(){
	test1();
	// test3();
}