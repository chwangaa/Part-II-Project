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

static inline void remove_matrix(Dtype* old_matrix){
    free(old_matrix);
}

#endif