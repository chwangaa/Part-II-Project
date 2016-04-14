#include "test_util.h"

const int M_default = 1024;
const int N_default = 1024;
const int K_default = 1024;
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
  double clk = benchmark_mm(M, N, K, NUM_REPEAT, ALGO);
  printf("%d, %d, %d, %f \n", M, N, K, clk);
}