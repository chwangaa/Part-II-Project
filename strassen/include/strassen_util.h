#ifndef STRASSEN_UTIL_H
#define STRASSEN_UTIL_H
#include "matrix.h"

#define limit_X 512
#define limit_M 512
#define limit_N 512
#define limit_K 512

#define MC 128
#define KC 384
#define NC 4096
#define MR 8
#define NR 8

static bool baseConditionReached(const unsigned int m,
                          const unsigned int n,
                          const unsigned int k){

    if(m <= limit_M || n <= limit_N || k < limit_K){
        return true;
    }
    else{
    	return false;
    }

}


#endif