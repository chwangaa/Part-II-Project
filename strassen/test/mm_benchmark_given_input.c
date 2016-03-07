#include "test_util.h"

const int M_default = 256;
const int N_default = 256;
const int K_default = 256;
const spacingFactor = 1;

int main(int argc, char** argv) {
  int M, N, K, ALGO;
  if (argc < 5) {
    fprintf(stderr, "M, N, K, ALGO not given, use the default values\n");
    M = M_default;
    N = N_default;
    K = K_default;
  }
  else{
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    ALGO = atoi(argv[4]);
  }

  int NUM_REPEAT = 1;
  benchmark_mm(M, N, K, NUM_REPEAT, ALGO);
}