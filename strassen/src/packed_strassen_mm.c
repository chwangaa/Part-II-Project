#include "strassen_util.h"
#include <emmintrin.h>
#include <immintrin.h>

// M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
static Dtype* strassen_make_M1_submatrix(    
    Dtype const *A_1_1, const int a11m, const int a11n, const int incRowA_1_1,
    Dtype const *A_2_2, const int a22m, const int a22n, const int incRowA_2_2,
    Dtype const *B_1_1, const int b11m, const int b11n, const int incRowB_1_1,
    Dtype const *B_2_2, const int b22m, const int b22n, const int incRowB_2_2){


    int M = a11m;
    int K = a11n;
    int N = b11n;
    Dtype* T1 = make_matrix(M, K);
    matrix_addition(M, K,
                    A_1_1, incRowA_1_1,
                    A_2_2, incRowA_2_2,
                    T1, K);

    Dtype* T2 = make_matrix(K, N);
    matrix_addition(K, N,
                    B_1_1, incRowB_1_1,
                    B_2_2, incRowB_2_2,
                    T2, N);    
    // M6 = T1 * T2  
    Dtype* M1 = make_matrix(M, N);
    packed_strassen_mm(
        M, N, K,
        T1, K,
        T2, N,
        M1, N);

    return M1;
}


static Dtype* strassen_make_M2_submatrix(
    Dtype const *A_2_1, const int a21m, const int a21n, const int incRowA_2_1,
    Dtype const *A_2_2, const int a22m, const int a22n, const int incRowA_2_2,
    Dtype const *B_1_1, const int b11m, const int b11n, const int incRowB_1_1){

    // sanity check
    debug_assert(a21m == a22m);
    debug_assert(a21n == b11m);
    debug_assert(a21n >= a22n);

    int M = a21m;
    int K = a21n;
    int N = b11n;
    /*
    construct M2 by the formula
    M2 = (A_2_1 + A_2_2) * B_1_1
    */
    // T1 = A_2_1 + A_2_2
    // Dtype* T1 = addDifferentSizedMatrix(A_2_1, m2, k1, incRowA_2_1,
    //                                     A_2_2, a22m, a22n, incRowA_2_2);
    Dtype* T1 = make_matrix(M, K);
    matrix_addition(M, K,
                    A_2_1, incRowA_2_1,
                    A_2_2, incRowA_2_2,
                    T1, K);

    Dtype* M2 = make_matrix(M, N);
    packed_strassen_mm(
        M, N, K,
        T1, K,
        B_1_1, incRowB_1_1,
        M2, N);

    remove_matrix(T1);
    return M2;
}


static Dtype* strassen_make_M3_submatrix(
    Dtype* const A_1_1, const int a11m, const int a11n, const int incRowA_1_1,
    Dtype* const B_1_2, const int b12m, const int b12n, const int incRowB_1_2,
    Dtype* const B_2_2, const int b22m, const int b22n, const int incRowB_2_2){


    // sanity check
    debug_assert(b12n == b22n);
    debug_assert(a11n == b12m);
    const int m1 = a11m;
    const int k1 = b12m;
    const int n1 = b12n;

    Dtype* T1 = subtractDifferentSizedMatrix(B_1_2, b12m, b12n, incRowB_1_2,
                                             B_2_2, b22m, b22n, incRowB_2_2);
    Dtype* M3 = make_matrix(m1, n1);
    
    packed_strassen_mm(
        m1, n1, k1,
        A_1_1, incRowA_1_1,
        T1, n1,
        M3, n1); 

    remove_matrix(T1);
    return M3;
}

static Dtype* strassen_make_M4_submatrix(    
    Dtype* const A_2_2, const int a22m, const int a22n, const int incRowA_2_2,
    Dtype* const B_2_1, const int b21m, const int b21n, const int incRowB_2_1,
    Dtype* const B_1_1, const int b11m, const int b11n, const int incRowB_1_1){

    // sanity check
    debug_assert(b21n == b11n);
    debug_assert(a22n <= b11m);
    int k2 = b21m;
    int n1 = b11n;
    int m2 = a22m;
    /*
    construct M4 by the formula
    M4 = A_2_2 * (B_1_1 - B_2_1)
    */
    Dtype* T1 = make_matrix(k2, n1);
    
    matrix_subtraction(k2, n1,
            B_1_1, incRowB_1_1,
            B_2_1, incRowB_2_1,
            T1, n1);

    Dtype* M4 = make_matrix(m2, n1);
    packed_strassen_mm(
        m2, n1, k2,
        A_2_2, incRowA_2_2,
        T1, n1,
        M4, n1);
    remove(T1);
    return M4;
}

static Dtype* strassen_make_M5_submatrix(
    Dtype* const A_1_1, const int a11m, const int a11n, const int incRowA_1_1,
    Dtype* const A_1_2, const int a12m, const int a12n, const int incRowA_1_2,
    Dtype* const B_2_2, const int b22m, const int b22n, const int incRowB_2_2){

    // sanity check
    debug_assert(a11m == a12m);
    debug_assert(a11n >= a12n);
    debug_assert(a11n >= b22m);
    const int m1 = a11m;
    const int k2 = b22m;
    const int n2 = b22n;

    Dtype* T1 = make_matrix(m1, k2);

    matrix_addition(m1, k2,
            A_1_1, incRowA_1_1,
            A_1_2, incRowA_1_2,
            T1,    k2);

    Dtype* M5 = make_matrix(m1, n2);
    packed_strassen_mm(
        m1, n2, b22m,
        T1, k2,
        B_2_2, incRowB_2_2,
        M5, n2);
    remove(T1);
    return M5;
}

// M6 = (A_1_1 - A_2_1) * (B_1_1 + B_1_2)
static Dtype* strassen_make_M6_submatrix(    
    Dtype const *A_2_1, const int a21m, const int a21n, const int incRowA_2_1,
    Dtype const *A_1_1, const int a11m, const int a11n, const int incRowA_1_1,
    Dtype const *B_1_1, const int b11m, const int b11n, const int incRowB_1_1,
    Dtype const *B_1_2, const int b12m, const int b12n, const int incRowB_1_2){

    // sanity check
    debug_assert(a11m >= a21m);
    debug_assert(a11n == a21n);
    debug_assert(b11m == b12m);
    debug_assert(b11n >= b12n);
    debug_assert(a11n == b11m);

    int M = a11m;
    int K = a11n;
    int N = b11n;

    Dtype* T1 = make_matrix(M, K);
    matrix_subtraction(M, K,
                        A_1_1, incRowA_1_1,
                        A_2_1, incRowA_2_1,
                        T1, K);
    // Dtype* T2 = addDifferentSizedMatrix(B_1_1, b11m, b11n, incRowB_1_1,
    //                                     B_1_2, b12m, b12n, incRowB_1_2);
    Dtype* T2 = make_matrix(K, N);
    matrix_addition(K, N,
                    B_1_1, incRowB_1_1,
                    B_1_2, incRowB_1_2,
                    T2, N);    
    // M6 = T1 * T2
    Dtype* M6 = make_matrix(M, N);
    packed_strassen_mm(
        M, N, K,
        T1, K,
        T2, N,
        M6, N);


    remove_matrix(T1);
    remove_matrix(T2);
    return M6;
}

// M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
static Dtype* strassen_make_M7_submatrix(
    Dtype const *A_1_2, int a12m, int a12n, const int incRowA_1_2,
    Dtype const *A_2_2, int a22m, int a22n, const int incRowA_2_2,
    Dtype const *B_2_1, int b21m, int b21n, const int incRowB_2_1,
    Dtype const *B_2_2, int b22m, int b22n, const int incRowB_2_2){


    // sanity check
    debug_assert(a12m >= a22m);
    debug_assert(a12n == a22n);
    debug_assert(b21m == b22m);
    debug_assert(b21n >= b22n);
    debug_assert(a12n == b21m);
    
    int M = a12m;
    int N = b21n;
    int K = a12n;

    Dtype* T1 = make_matrix(M, K);
    matrix_subtraction(M, K,
                    A_1_2, incRowA_1_2,
                    A_2_2, incRowA_2_2,
                    T1, K);

    Dtype* T2 = make_matrix(K, N);
    matrix_addition(K, N,
                    B_2_1, incRowB_2_1,
                    B_2_2, incRowB_2_2,
                    T2, N);

    Dtype* M7 = make_matrix(M, N);
    packed_strassen_mm(
        M, N, K,
        T1, K,
        T2, N,
        M7, N);
    remove_matrix(T1);
    remove_matrix(T2);
    return M7;
}

//  C_1_1 = M1 - M4 - M5 + M7
static void strassen_calculate_C_1_1(
    Dtype* const M1, int m1m, int m1n,
    Dtype* const M4, int m4m, int m4n,
    Dtype* const M5, int m5m, int m5n,
    Dtype* const M7, int m7m, int m7n,
    Dtype* C_1_1,    int M,   int N,
    int incRowC_1_1
    ){
    // C_1_1 = M1 - M4 - M5 + M7
    /*
    // C_1_1 = M1
    matrix_addition(m1m, m1n,
                  M1, m1n,
                  M7, m7n,
                  C_1_1, incRowC_1_1);
    // C_1_1 -= M4
    matrix_partial_subtraction(C_1_1, M, N, incRowC_1_1,
                  M4, m4m, m4n, m4n);
    // C_1_1 -= M5
    matrix_partial_subtraction(C_1_1, M, N, incRowC_1_1,
                  M5, m5m, m5n, m5n);
    */
    debug_assert(m1n == m4n);
    debug_assert(m5n == m7n);
    debug_assert(m4n == m7n);
    debug_assert(m7n == N);
    for(int i = 0; i < M; i += 1){
        Dtype* R1 = M1;
        Dtype* R2 = M4;
        Dtype* R3 = M5;
        Dtype* R4 = M7;
        for(int j = 0; j < N; j+= 8){
            __m256 r1 = _mm256_load_ps(R1+j);
            __m256 r2 = _mm256_load_ps(R2+j);
            __m256 r3 = _mm256_load_ps(R3+j);
            __m256 r4 = _mm256_load_ps(R4+j);
            __m256 temp = r1 - r2 - r3 + r4;         
            _mm256_store_ps(C_1_1+j, temp);
        }
        R1 += N;
        R2 += N;
        R3 += N;
        R4 += N;
        C_1_1 += incRowC_1_1;
    }
}

// C_1_2 = M3 + M5
static void strassen_calculate_C_1_2(
    Dtype* const M3, int m3m, int m3n, 
    Dtype* const M5, int m5m, int m5n,
    Dtype* C_1_2, int M, int N,
    int incRowC_1_2
    ){

    debug_assert(m3m == m5m);
    debug_assert(m3n == m5n);
    debug_assert(m3m == M);
    debug_assert(m3n == N);
    
    matrix_addition(M, N,
            M3, m3n,
            M5, m5n,
            C_1_2, incRowC_1_2);

}

// C_2_1 = M2 - M4
static void strassen_calculate_C_2_1(
    Dtype* const M2, int m2m, int m2n,
    Dtype* const M4, int m4m, int m4n,
    Dtype* C_2_1, int M, int N,
    int incRowC_2_1
    ){

    debug_assert(m2m == m4m);
    debug_assert(m2m == M);
    debug_assert(m2n == m4n);
    debug_assert(m2n == N);

    matrix_subtraction(M, N,
        M2, m2n,
        M4, m4n,
        C_2_1, incRowC_2_1);
}

//  C_2_2 = M1 - M2 + M3 - M6
static void strassen_calculate_C_2_2(
    Dtype* const M1, int m1m, int m1n,
    Dtype* const M2, int m2m, int m2n,
    Dtype* const M3, int m3m, int m3n,
    Dtype* const M6, int m6m, int m6n,
    Dtype* C_2_2,    int M,   int N,
    int incRowC_2_2
    ){
    /*
    const int m2 = m2m;
    const int n2 = m3n;

    // C_2_2 = M1 - M2
    matrix_subtraction(m2, n2,
            M1, m1n,
            M2, m2n,
            C_2_2, incRowC_2_2);
    
    // M1 += M3
    matrix_partial_addition(C_2_2, m2, n2, incRowC_2_2,
                  M3, m2, n2, m3n);
    
    // M1 -= M6
    matrix_partial_subtraction(C_2_2, m2, n2, incRowC_2_2,
                  M6, m2, n2, m6n);
    */
    debug_assert(m1n == m2n);
    debug_assert(m3n == m6n);
    debug_assert(m2n == m6n);
    debug_assert(m6n == N);
    for(int i = 0; i < M; i += 1){
        Dtype* R1 = M1;
        Dtype* R2 = M2;
        Dtype* R3 = M3;
        Dtype* R4 = M6;
        for(int j = 0; j < N; j+= 8){
            __m256 r1 = _mm256_load_ps(R1+j);
            __m256 r2 = _mm256_load_ps(R2+j);
            __m256 r3 = _mm256_load_ps(R3+j);
            __m256 r4 = _mm256_load_ps(R4+j);
            __m256 temp = r1 - r2 + r3 - r4;         
            _mm256_store_ps(C_2_2+j, temp);
        }
        R1 += N;
        R2 += N;
        R3 += N;
        R4 += N;
        C_2_2 += incRowC_2_2;
    }
}


/* this script only works for square matrices 
   where the length is a power of 2
*/
void packed_strassen_mm(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    Dtype const *A, const int incRowA,
    Dtype const *B, const int incRowB,
    Dtype *C, const int incRowC){
    // the matrices must have positive dimension
    debug_assert(m > 0);
    debug_assert(n > 0);
    debug_assert(k > 0);
    /* check if the base case has reached
       the base condition is defined as when all the dimensions are smaller than 256
    */
    // fprintf(stderr, "multiplying dimension m(%d), n(%d), k(%d) \n", m, n, k);

    if(baseConditionReached(m, n, k) || m % 8 != 0 || n % 8 != 0 || k % 8 != 0){
        packed_strassen_base_matrix_multiplication(
            m, n, k,
            A, incRowA,
            B, incRowB,
            C, incRowC);
        return;
    }
    /*
    We divide A, B, C into subsections as the following:
    A = |A_1_1, A_1_2|   B = |B_1_1, B_1_2|   C = |C_1_1, C_1_2|
        |A_2_1, A_2_2|       |B_2_1, B_2_2|       |C_2_1, C_2_2|


    We first calculate 7 temporary matrices as the follow:
    M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
    M2 = (A_2_1 + A_2_2) * B_1_1
    M3 = A_1_1 * (B_1_2 - B_2_2)
    M4 = A_2_2 * (B_1_1 - B_1_2)
    M5 = (A_1_1 + A_1_2) * B_2_2
    M6 = (A_1_1 - A_1_2) * (B_1_1 + B_1_2)
    M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)

    Then, we compute C section by section according to:
    C_1_1 = M1 - M4 - M5 + M7
    C_1_2 = M3 + M5
    C_2_1 = M2 - M4
    C_2_2 = M1 - M2 + M3 - M6 
    */
    const unsigned int m2 = m / 2;
    const unsigned int m1 = m - m2; 
    const unsigned int n2 = n / 2;
    const unsigned int n1 = n - n2;
    const unsigned int k2 = k / 2;
    const unsigned int k1 = k - k2;
    
    Dtype* const A_1_1 = A;
    Dtype* const A_1_2 = A_1_1 + k1 * m1;
    Dtype* const A_2_1 = A_1_2 + k2 * m1;
    Dtype* const A_2_2 = A_2_1 + k1 * m2;
    Dtype* C_1_1 = C;
    Dtype* C_1_2 = C_1_1 + n1;
    Dtype* C_2_1 = C_1_1 + incRowC*m1;
    Dtype* C_2_2 = C_2_1 + n1;

    Dtype* const B_1_1 = B;
    Dtype* const B_1_2 = B_1_1 + n1*k1;
    Dtype* const B_2_1 = B_1_2 + n2*k1;
    Dtype* const B_2_2 = B_2_1 + n1*k2;

    // this version assumes the dimension is divisible by 2



    /*
    construct M1 by the formula
    M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
    */
    Dtype* M1 = strassen_make_M1_submatrix(    
            A_1_1, m1, k1, k1,
            A_2_2, m2, k2, k2,
            B_1_1, k1, n1, n1,
            B_2_2, k2, n2, n2);
    
    /*
    construct M4 by the formula
    M4 = A_2_2 * (B_2_1 - B_1_1)
    */
    Dtype* M4 = strassen_make_M4_submatrix(
            A_2_2, m2, k2, k2,
            B_2_1, k2, n1, n1,
            B_1_1, k1, n1, n1);
    /*
    construct M5 by the formula
    M5 = (A_1_1 + A_1_2) * B_2_2
    */
    Dtype* M5 = strassen_make_M5_submatrix(
            A_1_1, m1, k1, k1,
            A_1_2, m1, k2, k2,
            B_2_2, k2, n2, n2);
    /*
    construct M7 by the formula
    M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
    */
    Dtype* M7 = strassen_make_M7_submatrix(
            A_1_2, m1, k2, k2,
            A_2_2, m2, k2, k2,
            B_2_1, k2, n1, n1,
            B_2_2, k2, n2, n2);
    /*!!!!!!!!!!!!!!!!!!!!!!!!!!
    compute C_1_1 by the formula
    C_1_1 = M1 + M4 - M5 + M7
    */
    strassen_calculate_C_1_1(
        M1, m1, n1,
        M4, m2, n1,
        M5, m1, n2,
        M7, m1, n1,
        C_1_1, m1, n1, incRowC);

    remove_matrix(M7);
    /*
    construct M2 by the formula
    M2 = (A_2_1 + A_2_2) * B_1_1
    */
    Dtype* M2 = strassen_make_M2_submatrix(
            A_2_1, m2, k1, k1,
            A_2_2, m2, k2, k2,
            B_1_1, k1, n1, n1);    
    /*!!!!!!!!!!!!!!!!!!!!!!!!!!
    compute C_2_1 by the formula
    C_2_1 = M2 + M4
    */
    strassen_calculate_C_2_1(
        M2, m2, n1,
        M4, m2, n1,
        C_2_1, m2, n1, incRowC);

    remove_matrix(M4);
    /*
    construct M3 by the formula
    M3 = A_1_1 * (B_1_2 - B_2_2)
    */
    Dtype* M3 = strassen_make_M3_submatrix(
            A_1_1, m1, k1, k1,
            B_1_2, k1, n2, n2,
            B_2_2, k2, n2, n2);
    /*!!!!!!!!!!!!!!!!!!!!!!!!!!
    compute C_1_2 by the formula
    C_1_2 = M3 + M5
    */
    strassen_calculate_C_1_2(
        M3, m1, n2,
        M5, m1, n2,
        C_1_2, m1, n2, incRowC);
    remove_matrix(M5);
    /*  
    construct M6 by the formula
    M6 = (A_1_1 - A_2_1) * (B_1_1 + B_1_2)
    */
    Dtype* M6 = strassen_make_M6_submatrix(
            A_2_1, m2, k1, k1,
            A_1_1, m2, k1, k1,
            B_1_1, k1, n2, n2,
            B_1_2, k1, n2, n2);
    /*!!!!!!!!!!!!!!!!!!!!!!!!!!
    compute C_2_2 by the formula
    C_2_2 = M1 - M2 + M3 + M6
    */
    
    strassen_calculate_C_2_2(
        M1, m1, n1,
        M2, m2, n1,
        M3, m1, n2,
        M6, m2, n2,
        C_2_2, m2, n2, incRowC);
    
    // print_matrix(M2, 47, 245, incRowC);
    // /*
    // remove the working space
    // */
    remove_matrix(M1);
    remove_matrix(M2);
    remove_matrix(M3);
    remove_matrix(M6);


    return;
}