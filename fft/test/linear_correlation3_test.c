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
	LinearCorrelation3(input, 3, 3, 1,
		  			   kernel, 2, 2, 1,
		  			   FFT_OVERWRITE,
		  			   output);

	FFT_Type correct[4] = {6,8,12,14};
	for(int i = 0; i < 4; i++){
		// print_complex(output[i]);
		assert(isEqual(output[i],correct[i]));
	}

	fprintf(stderr, "the first 3d linear correlation test passes\n");
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
	LinearCorrelation3(A, M, N, 1,
		  			   B, MB, NB, 1,
		  			   FFT_OVERWRITE,
		  			   C);
	for(int i = 0; i < 4; i++){
		assert(isEqual(C[i], Correct[i]));
		// print_complex(C[i]);
	}

	fprintf(stderr, "the second 3d linear correlation test passes\n");
}

/*
	multiple kernels test
*/
void test3(){
	// FFT_Type* A = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* B = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* C = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// assert(C);
	FFT_Type A[48] = {1,0,0,0,
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
	FFT_Type B[12] = {0,2,
	               3,0,
	               1,0,
	               0,1,
	               1,1,
	               1,1};
	FFT_Type C[9] = {0,0,0,0,0,0,0,0,0};

	FFT_Type Correct[9] = {11, 16, 15,
	                    21, 21, 26,
	                    27, 31, 31};
	
	LinearCorrelation3(A, 4, 4, 3,
		  			   B, 2, 2, 3,
		  			   FFT_OVERWRITE,
		  			   C);
	for(int i = 0; i < 9; i++){
		assert(isEqual(C[i], Correct[i]));
	}

	fprintf(stderr, "the third 3d linear correlation test passes\n");
}


void test4(){
	// FFT_Type* A = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* B = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* C = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// assert(C);
	FFT_Type A[48] = {1,0,0,0,
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
	FFT_Type B[12] = {0,2,
	               3,0,
	               1,0,
	               0,1,
	               1,1,
	               1,1};
	FFT_Type C[9] = {1,1,1,1,1,1,1,1,1};

	FFT_Type Correct[9] = {12, 17, 16,
	                    22, 22, 27,
	                    28, 32, 32};
	
	LinearCorrelation3(A, 4, 4, 3,
		  			   B, 2, 2, 3,
		  			   FFT_ACCUMULATE,
		  			   C);
	for(int i = 0; i < 9; i++){
		assert(isEqual(C[i], Correct[i]));
	}

	fprintf(stderr, "the fourth 3d linear correlation test passes\n");
}

void test5(){
	// FFT_Type* A = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* B = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* C = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// assert(C);
	FFT_Type A[16] = {1,0,0,0,
				   0,1,0,0,
				   0,0,1,0,
				   0,0,0,1};
	FFT_Type B[8] = {0,2,
	               	  3,0,
	               	  1,0,
	               	  0,1};
	FFT_Type C[9] = {0,0,0,0,0,0,0,0,0};
	FFT_Type A_correct[16] = {1,0,0,0,
				   0,1,0,0,
				   0,0,1,0,
				   0,0,0,1};
	FFT_Type Correct[9] = {2,3,0,
	                       2,2,3,
	                       0,2,2};
	
	LinearCorrelation3(A, 4, 4, 1,
		  			   B, 2, 2, 1,
		  			   FFT_ACCUMULATE,
		  			   C);
	LinearCorrelation3(A, 4, 4, 1,
		  			   &B[4], 2, 2, 1,
		  			   FFT_ACCUMULATE,
		  			   C);
	for(int i = 0; i < 9; i++){
		assert(isEqual(C[i], Correct[i]));
	}
	for(int i = 0; i < 16; i++){
		assert(isEqual(A[i], A_correct[i]));
	}
	fprintf(stderr, "the 5th 3d linear correlation test passes\n");
}


void test6(){
	// FFT_Type* A = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* B = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* C = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// assert(C);
	FFT_Type* A = (FFT_Type*)malloc(sizeof(FFT_Type*)*28*28);
	FFT_Type B[25] = {1,0,0,0,0,
					 0,1,0,0,0,
					 0,0,1,0,0,
					 0,0,0,1,0,
					 0,0,0,0,1};
	FFT_Type* C = (FFT_Type*)malloc(sizeof(FFT_Type*)*24*24);
	for(int i = 0; i < 28*28; i++){
		A[i] = i;
	}
	for(int i = 0; i < 24*24; i++){
		C[i] = 0;
	}
	LinearCorrelation3(A, 28, 28, 1,
		  			   B, 5, 5, 1,
		  			   FFT_ACCUMULATE,
		  			   C);

	for(int i = 0; i < 24; i++){
		// assert(isEqual(A[i], A_correct[i]));
		for(int j = 0; j < 24; j++){
			assert(isEqual(C[i*24+j], 290+i*140+j*5));
		}
	}
	fprintf(stderr, "the 6th 3d linear correlation test passes\n");
}

int main(){
	test1();
	test2();
	test3();
	test4();
	test5();
	test6();
}
