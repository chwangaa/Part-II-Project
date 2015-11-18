#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H
#include "setting.h"

Dtype* make_matrix(const unsigned int M, const unsigned int N){
    Dtype* new_matrix = (Dtype*)malloc(sizeof(Dtype)*M*N);
    assert(new_matrix);
    return new_matrix;
}

static inline void remove_matrix(Dtype* old_matrix){
    free(old_matrix);
}

#endif