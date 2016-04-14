#include "fft.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"


/*
	This test test whether the fft works properly
	input: [1,1,1,1,1,1,1,1,1,1]
	output: [10,0,0,0,0,0,0,0,0,0]
*/
void test1(){
	unsigned int M = 7;
	FFT_Type* input = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	FFT_Type correct[7] = {7, 0, 0, 0, 0, 0, 0};
	assert(input);
	assert(output);
	for(int i = 0; i < M; i++){
		input[i] = 1 + 0.0*I;
		output[i] = 0;
	}

	FFT(M, input, output);

	for(int i = 0; i < M; i++){
		assert(isEqual(output[i], correct[i]));
	}

	fprintf(stderr, "the first test passes\n");
	free(input);
	free(output);
}


/*
	This test test whether the fft works properly
	input: [0,1,2,3,4,5,6,7,8,9]
	output: [45, -5+1.53884177j, -5+6.88190960j
			 -5+3.63271264j, -5+1.62459848j, -5,
			 -5-1.62459848j, -5-3.63271264j, -5-6.88190960j,
			 -5-1.53884177j]
*/
void test2(){
	unsigned int M = 10;
	FFT_Type* input = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	FFT_Type correct[10] = {45, -5+15.3884177*I, -5+6.88190960*I,
			 -5+3.63271264*I, -5+1.62459848*I, -5,
			 -5-1.62459848*I, -5-3.63271264*I, -5-6.88190960*I,
			 -5-15.3884177*I};
	assert(input);
	assert(output);
	for(int i = 0; i < M; i++){
		input[i] = i;
		output[i] = 0;
	}

	FFT(M, input, output);

	for(int i = 0; i < M; i++){
		assert(isEqual(output[i], correct[i]));
	}

	fprintf(stderr, "the second test passes\n");
	free(input);
	free(output);
}

int main(){
	test1();
	test2();
	// test3();
	// test4();
	return 0;
}