#include "test_util.h"

void benchmark_vgg(){
    int NUM_TIMES = 3;
    int ALGO = 0;
    benchmark_mm(256, 3136, 1152, NUM_TIMES, ALGO);
    benchmark_mm(256, 3136, 2304, NUM_TIMES, ALGO);
    benchmark_mm(512, 784, 1152, NUM_TIMES, ALGO);
    benchmark_mm(512, 784, 2304, NUM_TIMES, ALGO);
    benchmark_mm(512, 784, 4608, NUM_TIMES, ALGO);
    benchmark_mm(512, 196, 4608, NUM_TIMES, ALGO);
}

void benchmark_square_mm(){
    int lower = 64;
    int upper = 4096;
    for(int i = lower; i <= upper; i *= 2){
        benchmark_mm(i, i, i, 3, 0);
    }
}

int main(){
    benchmark_square_mm();
}