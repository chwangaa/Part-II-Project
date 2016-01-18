#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>
#include "StrassenMatrixMultiplication.h"
// #include "src/matrix_util.h"
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

static void
pack_A_unit_strip(int k, const Dtype *A, int incRowA, int incColA,
          Dtype *buffer)
{
    int i, j;

    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }
        buffer += MR;
        A      += incColA;
    }
}

/*
 *  Packing panels from A
 */
static void
pack_A_strip(int mc, int kc, const Dtype *A, int incRowA, int incColA,
       Dtype *buffer)
{
    int mp  = mc / MR;
    int _mr = mc % MR;
    assert(_mr == 0);
    int i, j;

    for (i=0; i<mp; ++i) {
        pack_A_unit_strip(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A      += MR*incRowA;
    }
    // // needs to work with unit of 4, if not, pack with zero
    // if (_mr>0) {
    //     for (j=0; j<kc; ++j) {
    //         for (i=0; i<_mr; ++i) {
    //             buffer[i] = A[i*incRowA];
    //         }
    //         for (i=_mr; i<MR; ++i) {
    //             buffer[i] = 0.0;
    //         }
    //         buffer += MR;
    //         A      += incColA;
    //     }
    // }
}

/*
 *  Packing a complete micro panels from B
 */
static void
pack_B_unit_strip(int k, const Dtype *B, int incRowB, int incColB,
          Dtype *buffer)
{
    int i, j;

    for (i=0; i<k; ++i) {
        for (j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B      += incRowB;
    }
}

static void
pack_B_strip(int kc, int nc, const Dtype *B, int incRowB, int incColB,
       Dtype *buffer)
{
    int np  = nc / NR;
    int _nr = nc % NR;
    assert(_nr == 0);
    int i, j;

    for (j=0; j<np; ++j) {
        pack_B_unit_strip(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B      += NR*incColB;
    }
    // // needs to work in unit of 4, if not, pack with zero
    // if (_nr>0) {
    //     for (i=0; i<kc; ++i) {
    //         for (j=0; j<_nr; ++j) {
    //             buffer[j] = B[j*incColB];
    //         }
    //         for (j=_nr; j<NR; ++j) {
    //             buffer[j] = 0.0;
    //         }
    //         buffer += NR;
    //         B      += incRowB;
    //     }
    // }
}

static inline void MakePackedA(Dtype* A, int M, int K, int incRowA,
            int M_target, int K_target, 
            Dtype* newA){

    // make sure the dimension is okay
    assert(M % 8 == 0 && K % 8 == 0);
    // if the base case is reached, recurse down
    if(M == M_target && K == K_target){
        // pack_A_strip(M, K, A, incRowA, 1, newA);

            int mb = (M+MC-1) / MC;
            int kb = (K+KC-1) / KC;
            // fprintf(stderr, "kb is %d, k is %d \n ", kb, k);
            int _mc = M % MC;
            int _kc = K % KC;

            for (int l=0; l<kb; ++l) {
                int kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;

                for (int i=0; i<mb; ++i) {
                    int mc = (i!=mb-1 || _mc==0) ? MC : _mc;
                    pack_A_strip(mc, kc, &A[i*MC*incRowA+l*KC], incRowA, 1, newA);
                    newA += mc * kc;
            }
        }
        return;
    }
    // when base case is not reached, both dimension should be larger than the target
    assert(M > M_target && K > K_target);
    // find the head pointers of four submatrices
    const unsigned int m1 = M / 2;
    const unsigned int m2 = M - m1; 
    const unsigned int k1 = K / 2;
    const unsigned int k2 = K - k1;

    Dtype* A_1_1 = A;
    Dtype* A_1_2 = A_1_1 + k1;
    Dtype* A_2_1 = A_1_1 + incRowA*m1;
    Dtype* A_2_2 = A_2_1 + k1;

    Dtype* newA_1_1 = newA;
    Dtype* newA_1_2 = newA_1_1 + m1*k1;
    Dtype* newA_2_1 = newA_1_2 + m1*k2;
    Dtype* newA_2_2 = newA_2_1 + m2*k1; 

    MakePackedA(A_1_1, m1, k1, incRowA, M_target, K_target, newA_1_1);
    MakePackedA(A_1_2, m1, k2, incRowA, M_target, K_target, newA_1_2);
    MakePackedA(A_2_1, m2, k1, incRowA, M_target, K_target, newA_2_1);
    MakePackedA(A_2_2, m2, k2, incRowA, M_target, K_target, newA_2_2);
}

static inline void MakePackedB(Dtype* B, int K, int N, int incRowB,
            int K_target, int N_target, 
            Dtype* newB){

    // make sure the dimension is okay
    assert(newB);
    assert(N % 8 == 0 && K % 8 == 0);
    // if the base case is reached, recurse down
    if(K == K_target && N == N_target){
        pack_B_strip(K, N, B, incRowB, 1, newB);
        return;
    }
    // when base case is not reached, both dimension should be larger than the target
    assert(K > K_target && N > N_target);
    // find the head pointers of four submatrices
    const unsigned int n1 = N / 2;
    const unsigned int n2 = N - n1; 
    const unsigned int k1 = K / 2;
    const unsigned int k2 = K - k1;

    Dtype* B_1_1 = B;
    Dtype* B_1_2 = B_1_1 + n1;
    Dtype* B_2_1 = B_1_1 + incRowB*k1;
    Dtype* B_2_2 = B_2_1 + n1;

    Dtype* newB_1_1 = newB;
    Dtype* newB_1_2 = newB_1_1 + k1*n1;
    Dtype* newB_2_1 = newB_1_2 + k1*n2;
    Dtype* newB_2_2 = newB_2_1 + k2*n1; 

    MakePackedA(B_1_1, k1, n1, incRowB, K_target, N_target, newB_1_1);
    MakePackedA(B_1_2, k1, n2, incRowB, K_target, N_target, newB_1_2);
    MakePackedA(B_2_1, k2, n1, incRowB, K_target, N_target, newB_2_1);
    MakePackedA(B_2_2, k2, n2, incRowB, K_target, N_target, newB_2_2);
}


static int calculateMTarget(int M, int N, int K){
    while(M % 16 == 0 && N % 16 == 0 && K > limit_K && N > limit_N && M > limit_M){
        M /= 2;
        N /= 2;
        K /= 2;
    }
    return M;
}

static int calculateKTarget(int M, int N, int K){
    while(M % 16 == 0 && N % 16 == 0 && K > limit_K && N > limit_N && M > limit_M){
        M /= 2;
        N /= 2;
        K /= 2;
    }
    return K;
}

static int calculateNTarget(int M, int N, int K){
    while(M % 16 == 0 && N % 16 == 0 && K > limit_K && N > limit_N && M > limit_M){
        M /= 2;
        N /= 2;
        K /= 2;
    }
    return N;
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
    assert(A);
    assert(B);
    assert(C);
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A[i*incRowA+j] = i;
        }
    }
    
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            B[i*incRowB+j] = 1;
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
    Dtype* newA = (Dtype*)malloc_aligned4(32, sizeof(Dtype)*M*K);
    Dtype* newB = (Dtype*)malloc_aligned4(32, sizeof(Dtype)*K*N);
    
    int Mtarget = calculateMTarget(M, N, K);
    int Ktarget = calculateKTarget(M, N, K);
    int Ntarget = calculateNTarget(M, N, K);
    fprintf(stderr, "the targets are %d %d %d \n", Mtarget, Ntarget, Ktarget);
    MakePackedA(A, M, K, incRowA, 
                Mtarget, Ktarget,
                newA);
    MakePackedB(B, K, N, incRowB,
                Ktarget, Ntarget,
                newB);
    // free(A);
    // for(int i = 0; i < 16; i++){
    //     for(int j = 0; j < 16; j++){
    //         fprintf(stderr, "%d ", (int)newA[i*16+j]);
    //     }
    //     fprintf(stderr, "\n");
    // }

    uint64_t start_time = timestamp_us();
    strassen_matrix_multiplication(
            M, N, K,
            newA, incRowA,
            newB, incRowB,
            C, incRowC);

    uint64_t end_time = timestamp_us();
    double m_second_taken = (double)(end_time - start_time) / 1000.0;
    int error = 0;
    for(int i = 0; i < M; i++){
    // 	// fprintf(stderr, "%d \n", fix16_to_int(M3[i]));
    	for(int j = 0; j < N; j++){
            if(C[i*incRowC+j] != K*i){
                error++;
                // fprintf(stderr, "%d ", (int)C[i*incRowC+j]);

                // fprintf(stderr, "%d %d %f \n", i, j, C[i*incRowC+j]);
            }
    		    // fprintf(stderr, "%d ", (int)C[i*incRowC+j]);
            }
        // fprintf(stderr, "\n");
    }
    free_aligned4(newA);
    free_aligned4(newB);
    printf("%d, %d, %d, %d, %f \n", M, N, K, error, m_second_taken);
   // free(A);
    // free(B);
    // free(C);
}