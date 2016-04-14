#ifndef STRASSEN_UTIL_H
#define STRASSEN_UTIL_H
#include "matrix.h"

#define limit_X 256
#define limit_M 256
#define limit_N 256
#define limit_K 256

#define MC 128
#define KC 256
#define NC 4096
#define MR 8
#define NR 8

static bool baseConditionReached(const unsigned int m,
                          const unsigned int n,
                          const unsigned int k){

    if(m <= limit_M || n <= limit_N || k <= limit_K){
        return true;
    }
    else{
    	return false;
    }

}

static bool packedBaseConditionReached(const unsigned int m,
							const unsigned int n,
							const unsigned int k){
	if(m < MC*2 || n < NC*2 || k < KC){
		return true;
	}
	else{
		return false;
	}
}

#endif