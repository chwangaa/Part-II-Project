#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H
#include "setting.h"

Dtype* make_matrix(const unsigned int M, const unsigned int N){
    Dtype* new_matrix = (Dtype*)malloc(sizeof(Dtype)*M*N);
    assert(new_matrix);
    return new_matrix;
}

Dtype* pad_matrix(Dtype* old_matrix, const unsigned int old_M, const unsigned int old_N,
				  const unsigned int old_incRow,
				  const unsigned int new_M,
				  const unsigned int new_N){

	assert(new_M >= old_M);
	assert(new_N >= old_N);
	Dtype* new_matrix = make_matrix(new_M, new_N);

	for(int i = 0; i < new_M; i++){
		for(int j = 0; j < new_N; j++){
			new_matrix[i*new_N + j] = 0;
		}
	}

	for(int i = 0; i < old_M; i++){
		for(int j = 0; j < old_N; j++){
			new_matrix[i*new_N + j] = old_matrix[i*old_incRow + j];
		}
	}

	return new_matrix;
}


int getNumberLargerThanXAndIsPowerOfTwo(unsigned int X){
	int result = 1;
	int value = X;
	while(X > 1){
		X /= 2;
		result *= 2;
	}
	if(result >= value){
		return result;
	}
	else{
		return result * 2;
	}
}

int max(int a, int b){
	if(a > b){
		return a;
	}
	else{
		return b;
	}
}

int maxThree(int a, int b, int c){
	int max = -100000;
	if(a > max){
		max = a;
	}
	if(b > max){
		max = b;
	}
	if(c > max){
		max = c;
	}
	return max;
}
Dtype* padMatrixToPowerSquareMatrix(Dtype* old_matrix, const unsigned int M, const unsigned int N,
									const unsigned incRow){
	int side_length = max(M, N);
	int new_length = getNumberLargerThanXAndIsPowerOfTwo(side_length);
	return pad_matrix(old_matrix, M, N, incRow, new_length, new_length);
}

static inline void remove_matrix(Dtype* old_matrix){
    free(old_matrix);
}

static inline void print_matrix(Dtype* matrix, int M, int N, int incRow){
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
			fprintf(stderr, "%d ", (int)matrix[i*incRow+j]);
		}
		fprintf(stderr, "\n");
	}
}

static inline void matrix_copyTo(
	Dtype* from_matrix, int M, int N, int incRowFrom,
	Dtype* to_matrix, int M_to, int N_to, int incRowTo){

	assert(M_to <= M);
	assert(N_to <= N);
	for(int i = 0; i < M_to; i++){
		for(int j = 0; j < N_to; j++){
			to_matrix[i*incRowTo+j] = from_matrix[i*incRowFrom+j];
		}
	}
}
#endif