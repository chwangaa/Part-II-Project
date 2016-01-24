#include "fft.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "fft_util.h"


/*
	This test whether ifft2(fft(A)*fft(B)) returns the A(*)B
	A: [[1,2,3],[4,5,6],[7,8,9]]
	B: [[1,1,0],[1,1,0],[1,1,0]]
	C: [[20,18,22],[14,12,16],[26,24,28]]
*/
void test1(){
	unsigned int M = 3;
	unsigned int N = 3;
	// FFT_Type* A = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* B = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// FFT_Type* C = (FFT_Type*)malloc(sizeof(FFT_Type*)*M*N);
	// assert(C);
	FFT_Type A[9] = {1,2,3,4,5,6,7,8,9};
	FFT_Type B[9] = {1,1,0,1,1,0,0,0,0};
	FFT_Type C[9] = {0,0,0,0,0,0,0,0,0};
	FFT_Type Correct[9] = {20,18,22,14,12,16,26,24,28};
	CONV2(A, M, N,
		  B, M, N,
		  0,
		  C);

	for(int i = 0; i < M*N; i++){
		assert(isEqual(C[i], Correct[i]));
	}

	fprintf(stderr, "the first conv2d test passes\n");
}

/*
	This test whether ifft2(fft(A)*fft(B)) returns the A(*)B
	A: [[1,2,3],[4,5,6],[7,8,9]]
	B: [[1,1],[1,1]]
	C: [[12,16],[24,28]]
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
	CONV2(A, M, N,
		  B, MB, NB,
		  0,
		  C);
	for(int i = 0; i < 4; i++){
		assert(isEqual(C[i], Correct[i]));
		// print_complex(C[i]);
	}

	fprintf(stderr, "the second conv2d test passes\n");
}

int main(){
	test1();
	test2();
	return 0;
}