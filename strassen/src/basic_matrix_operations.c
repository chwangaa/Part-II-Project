#include "matrix.h"
#include "strassen_util.h"

#include <emmintrin.h>
#include <immintrin.h>

/*
    vector addition, of length N
*/
void add_row(Dtype* A, Dtype* B, Dtype* C, int N){
    
    int vector_size = (N / 8) * 8;
    int j = 0;
    for(; j < vector_size; j+=8){
        __m256 r1 = _mm256_loadu_ps(A+j);
        __m256 temp = _mm256_loadu_ps(&B[j]);
        temp += r1;         
        _mm256_storeu_ps(&C[j], temp);
    }

    for(; j < N; j++){
        C[j] = A[j] + B[j];
    }
}

/*
    vector subtraction, of length N
*/
void subtract_row(Dtype* A, Dtype* B, Dtype* C, int N){
    int vector_size = (N / 8) * 8;
    int j = 0;
    for(; j < vector_size; j+=8){
        __m256 r1 = _mm256_loadu_ps(A+j);
        __m256 temp = _mm256_loadu_ps(&B[j]);
        temp = r1 - temp;         
        _mm256_storeu_ps(&C[j], temp);
    }

    for(; j < N; j++){
        C[j] = A[j] - B[j];
    }   
}

void copy_row(Dtype* A, Dtype* B, int N){
    int vector_size = (N / 8) * 8;
    int j = 0;
    for(; j < vector_size; j+=8){
        __m256 r1 = _mm256_loadu_ps(B+j);
        _mm256_storeu_ps(&A[j], r1);
    }

    for(; j < N; j++){
        A[j] = B[j];
    }
}

void matrix_addition(
    const unsigned int M,
    const unsigned int N,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

    Dtype* A_base;
    Dtype* B_base;
    Dtype* C_base;
    for(int i = 0; i < M; i++){
        add_row(&A[i*incRowA], &B[i*incRowB], &C[i*incRowC], N);
    }
}

void matrix_subtraction(
    const unsigned int M,
    const unsigned int N,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

    for(int i = 0; i < M; i++){
        subtract_row(&A[i*incRowA], &B[i*incRowB], &C[i*incRowC], N);
    }
}

Dtype* make_matrix(const unsigned int M, const unsigned int N){
    // Dtype* new_matrix = (Dtype*)malloc(sizeof(Dtype)*M*N);
    Dtype* new_matrix = (Dtype*)_mm_malloc(sizeof(Dtype)*M*N,32);
    assert(new_matrix);
    return new_matrix;
}

Dtype* create_matrix(int M, int N, int incRow, Dtype default_value){
    Dtype* matrix = make_matrix(M, incRow);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            matrix[i*incRow+j] = default_value;
        }
    }

    return matrix;
}

Dtype* pad_matrix(Dtype* old_matrix, const unsigned int old_M, const unsigned int old_N,
                  const unsigned int old_incRow,
                  const unsigned int new_M,
                  const unsigned int new_N){
    assert(new_M >= old_M);
    assert(new_N >= old_N);
    Dtype* new_matrix = make_matrix(new_M, new_N);
    assert(new_matrix);
    
    for(int i = 0; i < new_M; i++){
        for(int j = 0; j < new_N; j++){
            new_matrix[i*new_N + j] = 0;
        }
    }

    for(int i = 0; i < old_M; i++){
        copy_row(&new_matrix[i*new_N], &old_matrix[i*old_incRow], old_N);
    }

    return new_matrix;
}


void remove_matrix(Dtype* old_matrix){
    _mm_free(old_matrix);
}

void print_matrix(Dtype* matrix, int M, int N, int incRow){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            fprintf(stderr, "%.1f ", matrix[i*incRow+j]);
        }
        fprintf(stderr, "\n");
    }
}

void matrix_copyTo(
    Dtype* const from_matrix, int M, int N, int incRowFrom,
    Dtype* to_matrix, int M_to, int N_to, int incRowTo){

    assert(M_to <= M);
    assert(N_to <= N);
    for(int i = 0; i < M_to; i++){
        // memcpy(&to_matrix[i*incRowTo], &from_matrix[i*incRowFrom], sizeof(Dtype)*N_to);
        copy_row(&to_matrix[i*incRowTo], &from_matrix[i*incRowFrom], N_to);         
    }
}

Dtype* matrix_copy(
    Dtype* const from_matrix, int M, int N, int incRowFrom){

    Dtype* new_matrix = make_matrix(M, N);
    for(int i = 0; i < M; i++){
        // memcpy(&new_matrix[i*N], &from_matrix[i*incRowFrom], sizeof(Dtype)*N);
        copy_row(&new_matrix[i*N], &from_matrix[i*incRowFrom], N);

    }
    return new_matrix;
}

void matrix_partial_addition(Dtype* result, int rM, int rN, int rincRow,
                             const Dtype* adder,  int aM, int aN, int aincRow){
    debug_assert(aM <= rM);
    debug_assert(aN <= rN);
    for(int i = 0; i < aM; i++){
        add_row(&result[i*rincRow], &adder[i*aincRow], &result[i*rincRow], aN);
    }
}

void matrix_partial_subtraction(Dtype* result, int rM, int rN, int rincRow,
                             const Dtype* adder,  int aM, int aN, int aincRow){
    debug_assert(aM <= rM);
    debug_assert(aN <= rN);
    for(int i = 0; i < aM; i++){
        subtract_row(&result[i*rincRow], &adder[i*aincRow], &result[i*rincRow], aN);
    }
}

Dtype* addDifferentSizedMatrix(
    Dtype* const Larger, int lm, int ln, int incRowL,
    Dtype* const Smaller, int sm, int sn, int incRowS){

    Dtype* new_matrix = matrix_copy(Larger, lm, ln, incRowL);
    matrix_partial_addition(new_matrix, lm, ln, ln,
                            Smaller, sm, sn, incRowS);
    return new_matrix;
}

Dtype* subtractDifferentSizedMatrix(
    Dtype* const Larger, int lm, int ln, int incRowL,
    Dtype* Smaller, int sm, int sn, int incRowS){

    Dtype* new_matrix = matrix_copy(Larger, lm, ln, incRowL);
    assert(new_matrix);
    matrix_partial_subtraction(new_matrix, lm, ln, ln,
                            Smaller, sm, sn, incRowS);
    return new_matrix;
}


static void
pack_A_unit_strip(int k, const Dtype *A, int incRowA, int incColA,
          Dtype *buffer)
{
    int i, j;


    for (j=0; j<k; ++j) {
        for (i=0; i<MR; ++i) {
            // fprintf(stderr, "assigning %f to %f \n", buffer[i], A[i*incRowA]);
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
}

void MakePackedA(Dtype* A, int M, int K, int incRowA,
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
            // fprintf(stderr, "K is %d, KC is %d \n", K, KC);
            // fprintf(stderr, "_mc is %d, _kc is %d \n", _mc, _kc);

            for (int l=0; l<kb; ++l) {
                int kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;

                for (int i=0; i<mb; ++i) {
                    int mc = (i!=mb-1 || _mc==0) ? MC : _mc;
                    // fprintf(stderr, "pack a strip of size %d %d \n", mc, kc);
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

void MakePackedB(Dtype* B, int K, int N, int incRowB,
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
