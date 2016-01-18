#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include "StrassenMatrixMultiplication.h"
#include "util.h"
#include "setting.h"
const unsigned int M_default = 16;
const unsigned int N_default = 16;
const unsigned int K_default = 16;
const unsigned int spacingFactor = 1;

void *malloc_aligned4(size_t alignment, size_t bytes)
{
    // we need to allocate enough storage for the requested bytes, some 
    // book-keeping (to store the location returned by malloc) and some extra
    // padding to allow us to find an aligned byte.  im not entirely sure if 
    // 2 * alignment is enough here, its just a guess.
    const size_t total_size = bytes + (2 * alignment) + sizeof(size_t);

    // use malloc to allocate the memory.
    char *data = malloc(sizeof(char) * total_size);

    if (data)
    {
        // store the original start of the malloc'd data.
        const void * const data_start = data;

        // dedicate enough space to the book-keeping.
        data += sizeof(size_t);

        // find a memory location with correct alignment.  the alignment minus 
        // the remainder of this mod operation is how many bytes forward we need 
        // to move to find an aligned byte.
        const size_t offset = alignment - (((size_t)data) % alignment);

        // set data to the aligned memory.
        data += offset;

        // write the book-keeping.
        size_t *book_keeping = (size_t*)(data - sizeof(size_t));
        *book_keeping = (size_t)data_start;
    }

    return data;
}

void free_aligned4(void *raw_data)
{
    if (raw_data)
    {
        char *data = raw_data;

        // we have to assume this memory was allocated with malloc_aligned.  
        // this means the sizeof(size_t) bytes before data are the book-keeping 
        // which points to the location we need to pass to free.
        data -= sizeof(size_t);

        // set data to the location stored in book-keeping.
        data = (char*)(*((size_t*)data));

        // free the memory.
        free(data);
    }
}

int main(int argc, char** argv) {
  int M, N, K;
  if (argc < 4) {
    fprintf(stderr, "M, N, K not given, use the default values\n");
    M = M_default;
    N = N_default;
    K = K_default;
  }
  else{
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }
    int incRowA = K * spacingFactor;
    int incRowB = N * spacingFactor;
    int incRowC = N * spacingFactor;

	Dtype* A = (Dtype*)malloc_aligned4(32, sizeof(Dtype)*M*incRowA);
	Dtype* B = (Dtype*)malloc_aligned4(32, sizeof(Dtype)*K*incRowB);
	Dtype* C = (Dtype*)malloc_aligned4(32, sizeof(Dtype)*M*incRowC);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A[i*incRowA+j] = 1;
        }
    }
    
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            B[i*incRowB+j] = j;
        }
    }

    
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            C[i*incRowC+j] = 0;
        }
    }


    // float temp[9] = {2, 2, 1, 2, 2, 1, 1, 1, 1};
    // float temp1[2] = {2,1};
    // float temp2[4] = {2, 2, 2, 2};
    // A = temp1;
    // B = temp2;

    uint64_t start_time = timestamp_us();
    strassen_matrix_multiplication(
            M, N, K,
            A, incRowA,
            B, incRowB,
            C, incRowC);
    
    uint64_t end_time = timestamp_us();
    double m_second_taken = (double)(end_time - start_time) / 1000.0;
    int error = 0;
    for(int i = 0; i < M; i++){
    // 	// fprintf(stderr, "%d \n", fix16_to_int(M3[i]));
    	for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != K*j){
                error++;
                // fprintf(stderr, "%d %d %f \n", i, j, C[i*incRowC+j]);
            }
    		    // fprintf(stderr, "%d ", (int)C[i*incRowC+j]);
            }
        // fprintf(stderr, "\n");
    }
    printf("%d, %d, %d, %d, %f \n", M, N, K, error, m_second_taken);
}