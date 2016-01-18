#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
// #include "SimpleMatrixMultiplication.h"
#include "matrix/matrix.h"
// #include "CacheObliviousMatrixMultiplication.h"
// #include "StrassenMatrixMultiplication.h"
#include "util.h"
const unsigned int M_default = 2;
const unsigned int N_default = 2;

int main(int argc, char** argv) {
  int M, N, new_M, new_N;
  if (argc < 5) {
    fprintf(stderr, "M, N not given, use the default values\n");
    M = M_default;
    N = N_default;
    new_M = M*2;
    new_N = N*2;
  }
  else{
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    new_M = atoi(argv[3]);
    new_N = atoi(argv[4]);
  }
  fprintf(stderr, "the values are: M(%d); N(%d); new_M(%d); new_N(%d)", M, N, new_M, new_N);
    int incRowA = N;

	Dtype* A = (Dtype*)malloc(sizeof(int)*M*incRowA);


    // initialize the original matrix to all 1s
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            A[i*incRowA+j] = j;
        }
    }

    // print the initial matrix
    fprintf(stderr, "printing the original matrix \n");
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            fprintf(stderr, "%d", (int)A[i*N]);
        }
        fprintf(stderr, "\n");
    }

    Dtype* new_A = padMatrixToPowerSquareMatrix(A, M, N, incRowA);
    int longer_side = max(M, N);
    new_M = new_N = getNumberLargerThanXAndIsPowerOfTwo(longer_side);
    fprintf(stderr, "print the new matrix after zero padding \n");
    for(int i = 0; i < new_M; i++){
        for(int j = 0; j < new_N; j++){
            fprintf(stderr, "%d", (int)(new_A[i*new_N + j]));
        }
        fprintf(stderr, "\n");
    }
}