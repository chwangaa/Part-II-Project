#include "fft.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"


/*
	This test test whether the fft works properly
	input: [[1,1,1],[1,1,1],[1,1,1]]
	output: [[9,0,0],[0,0,0],[0,0,0]]
*/
void test1(){
	unsigned int M = 3;
	unsigned int N = 3;
	FFT_Type* input = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
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
	assert(isEqual(output[0],9.0+0.0*I));

	for(int i = 1; i < M*N; i++){
		assert(isEqual(output[i], 0.0+0*I));
	}

	fprintf(stderr, "the first test passes\n");
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
void test2(){
	unsigned int M = 3;
	unsigned int N = 3;
	FFT_Type* input = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M*N; i++){
			input[i] = i + 1;
	}

	FFT2(M, N,
		input,
		output);
		assert(isEqual(output[0],45.0 + 0*I));
		assert(isEqual(output[1],-4.5+2.5980762*I));
		assert(isEqual(output[2],-4.5-2.5980762*I));
		assert(isEqual(output[3],-13.5+7.7942286*I));
		assert(isEqual(output[4],0+0*I));
		assert(isEqual(output[5],0+0*I));
		assert(isEqual(output[6],-13.5-7.7942286*I));
		assert(isEqual(output[7],0+0*I));
		assert(isEqual(output[8],0+0*I));

	fprintf(stderr, "the second test passes\n");
	free(input);
	free(output);
}

/*
	input: [[1,2,3],[4,5,6],[7,8,9,],[10,11,12]]
	output:[[1,2,3],[4,5,6],[7,8,9,],[10,11,12]]
*/
void test3(){
	unsigned int M = 1;
	unsigned int N = 2;
	FFT_Type* input = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	FFT_Type* output = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	assert(input);
	assert(output);

	for(int i = 0; i < M * N; i++){
		input[i] = i;
	}

	FFT2(M, N,
		input,
		output);

	assert(isEqual(output[0], 1));
	assert(isEqual(output[1], -1));

	fprintf(stderr, "the third test passes\n");
	free(input);
	free(output);
}

int main(){
	test1();
	test2();
	test3();
	return 0;
}