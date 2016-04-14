#include "fft.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"


/*
	input: [10,0,0,0,0,0,0,0,0,0]
	output: [1,1,1,1,1,1,1,1,1,1]
*/
void test1(){
	unsigned int M = 10;
	FFT_Type* input = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	FFT_Type correct[10] = {1,1,1,1,1,1,1,1,1,1};
	assert(input);
	assert(output);
	for(int i = 0; i < M; i++){
		input[i] = 0;
	}
	input[0] = 10;

	IFFT(M, input, output);

	for(int i = 0; i < M; i++){
		assert(isEqual(output[i], correct[i]));
	}

	fprintf(stderr, "the first test passes\n");
	free(output);
}


/*
	input: [45, -5+1.53884177j, -5+6.88190960j
			 -5+3.63271264j, -5+1.62459848j, -5,
			 -5-1.62459848j, -5-3.63271264j, -5-6.88190960j,
			 -5-1.53884177j]
	output: [0,1,2,3,4,5,6,7,8,9]
*/
void test2(){
	unsigned int M = 10;
	FFT_Type input[10] = {45, -5+15.3884177*I, -5+6.88190960*I,
			 -5+3.63271264*I, -5+1.62459848*I, -5,
			 -5-1.62459848*I, -5-3.63271264*I, -5-6.88190960*I,
			 -5-15.3884177*I};
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	assert(output);

	IFFT(M, input, output);

	for(int i = 0; i < M; i++){
		assert(isEqual(output[i], i));
	}

	fprintf(stderr, "the second test passes\n");
	free(output);
}

void test3(){
	int M = 500;
	FFT_Type* input = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M);
	assert(output);

	for(int i = 0; i < M; i++){
		input[i] = (float)i / 10;
	}

	FFT(M, input, output);
	IFFT(M, output, input);

	for(int i = 0; i < M; i++){
		assert(isEqual(input[i], (float)i / 10));	
	}


	fprintf(stderr, "the third test passes\n");
	free(output);
}

int main(){
	test1();
	test2();
	test3();
	// test4();
	return 0;
}