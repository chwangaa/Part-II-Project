#include "test_util.h"

const int M_default = 256;
const int N_default = 256;

int main(int argc, char** argv) {
  int M, N, Repeat;
  if (argc < 4) {
    fprintf(stderr, "M, N not given, use the default values\n");
    M = M_default;
    N = N_default;
    Repeat = 3;
  }
  else{
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    Repeat = atoi(argv[3]);
  }

  double m = benchmark_matrix_addition(M, N, Repeat);
  fprintf(stderr, "matrix addition of size %d, %d, takes %lf milisecond \n", M, N, m);
}