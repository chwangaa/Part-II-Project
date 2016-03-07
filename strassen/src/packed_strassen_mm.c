#include "strassen_util.h"


// M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
static Dtype* strassen_make_M1_submatrix(    
    Dtype const *A_1_1, const int a11m, const int a11n, const int incRowA_1_1,
    Dtype const *A_2_2, const int a22m, const int a22n, const int incRowA_2_2,
    Dtype const *B_1_1, const int b11m, const int b11n, const int incRowB_1_1,
    Dtype const *B_2_2, const int b22m, const int b22n, const int incRowB_2_2){

    /*
    construct M1 by the formula
    M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
    */
    // T1 = (A_1_1 + A_2_2)
    // Dtype* T1 = make_matrix(a11m, a11n);
    // // T1 = A_1_1
    // matrix_copyTo(A_1_1, a11m, a11n, incRowA_1_1,
    //               T1, a11m, a11n, a11n);

    // // T1 += A_2_2
    // matrix_partial_addition(
    //     T1, a11m, a11n, a11n,
    //     A_2_2, a22m, a22n, incRowA_2_2
    //     );
    Dtype* T1 = addDifferentSizedMatrix(A_1_1, a11m, a11n, incRowA_1_1,
                                        A_2_2, a22m, a22n, incRowA_2_2);
    // print_matrix(A_1_1, a11m, a11n, a11n);
    // T2 = (B_1_1 + B_2_2)

    // T2 = B_1_1
    // Dtype* T2 = make_matrix(b11m, b11n);

    // matrix_copyTo(B_1_1, b11m, b11n, incRowB_1_1,
    //               T2, b11m, b11n, b11n);
    // // T2 += B_2_2
    // matrix_partial_addition(
    //     T2, b11m, b11n, b11n,
    //     B_2_2, b22m, b22n, incRowB_2_2
    //     );
    Dtype* T2 = addDifferentSizedMatrix(B_1_1, b11m, b11n, incRowB_1_1,
                                        B_2_2, b22m, b22n, incRowB_2_2);
    // M1 = T1 * T2
    int m = a11m;
    int n = b11n;
    int k = a11n;
    Dtype* M1 = make_matrix(m, n);
    packed_strassen_mm(
        m, n, k,
        T1, k,
        T2, n,
        M1, n);
    remove_matrix(T1);
    remove_matrix(T2);
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

    int m2 = a21m;
    int k1 = a21n;
    int n1 = b11n;
    /*
    construct M2 by the formula
    M2 = (A_2_1 + A_2_2) * B_1_1
    */
    // T1 = A_2_1 + A_2_2
    /*
    Dtype* T1 = make_matrix(m2, k1);
    // T1 = A_2_1
    matrix_copyTo(A_2_1, m2, k1, incRowA_2_1,
                  T1, m2, k1, k1);
    // T1 += A_2_2
    matrix_partial_addition(
        T1, m2, k1, k1,
        A_2_2, a22m, a22n, incRowA_2_2
        );
    */
    Dtype* T1 = addDifferentSizedMatrix(A_2_1, m2, k1, incRowA_2_1,
                                        A_2_2, a22m, a22n, incRowA_2_2);
    // print_matrix(B_1_1, m2, n1, n1);
    Dtype* M2 = make_matrix(m2, n1);
    packed_strassen_mm(
        m2, n1, k1,
        T1, k1,
        B_1_1, incRowB_1_1,
        M2, n1);
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
    const int m1 = a11m;
    const int k1 = a11n;
    const int n1 = b11n;

    Dtype* T1 = subtractDifferentSizedMatrix(A_1_1, a11m, a11n, incRowA_1_1,
                                             A_2_1, a21m, a21n, incRowA_2_1);
    Dtype* T2 = addDifferentSizedMatrix(B_1_1, b11m, b11n, incRowB_1_1,
                                        B_1_2, b12m, b12n, incRowB_1_2);

    // M6 = T1 * T2
    Dtype* M6 = make_matrix(m1, n1);
    packed_strassen_mm(
        m1, n1, k1,
        T1, k1,
        T2, n1,
        M6, n1);

    remove_matrix(T1);
    remove_matrix(T2);
    return M6;
}

// M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
static void strassen_make_M7_submatrix(
    Dtype const *A_1_2, int a12m, int a12n, const int incRowA_1_2,
    Dtype const *A_2_2, int a22m, int a22n, const int incRowA_2_2,
    Dtype const *B_2_1, int b21m, int b21n, const int incRowB_2_1,
    Dtype const *B_2_2, int b22m, int b22n, const int incRowB_2_2,
    Dtype* result, int incRowResult){


    // sanity check
    debug_assert(a12m >= a22m);
    debug_assert(a12n == a22n);
    debug_assert(b21m == b22m);
    debug_assert(b21n >= b22n);
    debug_assert(a12n == b21m);
    
    int m1 = a12m;
    int n1 = b21n;
    int k2 = a12n;
    Dtype* T1 = subtractDifferentSizedMatrix(A_1_2, a12m, a12n, incRowA_1_2,
                                             A_2_2, a22m, a22n, incRowA_2_2);    
    Dtype* T2 = addDifferentSizedMatrix(B_2_1, b21m, b21n, incRowB_2_1,
                                        B_2_2, b22m, b22n, incRowB_2_2);
    // M7 = T1 * T2
    // Dtype* M7 = make_matrix(m1, n1);
    packed_strassen_mm(
        m1, n1, k2,
        T1, k2,
        T2, n1,
        result, incRowResult);

    remove_matrix(T1);
    remove_matrix(T2);
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

    if(baseConditionReached(m, n, k)){
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
    construct M7 by the formula
    M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
    */
    strassen_make_M7_submatrix(
            A_1_2, m1, k2, k2,
            A_2_2, m2, k2, k2,
            B_2_1, k2, n1, n1,
            B_2_2, k2, n2, n2,
            C_1_1, incRowC);
    
    /*
    construct M1 by the formula
    M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
    */    
    Dtype* M1 = strassen_make_M1_submatrix(    
            A_1_1, m1, k1, k1,
            A_2_2, m2, k2, k2,
            B_1_1, k1, n1, n1,
            B_2_2, k2, n2, n2);
    matrix_partial_addition(C_1_1, m1, n1, incRowC,
                  M1, m1, n1, n1);
    matrix_copyTo(M1, m1, n1, n1,
                  C_2_2, m2, n2, incRowC);
    remove_matrix(M1);
    /*
    construct M2 by the formula
    M2 = (A_2_1 + A_2_2) * B_1_1
    */
    Dtype* M2 = strassen_make_M2_submatrix(
            A_2_1, m2, k1, k1,
            A_2_2, m2, k2, k2,
            B_1_1, k1, n1, n1);    

    matrix_copyTo(M2, m2, n1, n1,
                  C_2_1, m2, n1, incRowC);
    matrix_partial_subtraction(C_2_2, m2, n2, incRowC,
                  M2, m2, n2, n1);
    remove_matrix(M2);
    /*
    construct M3 by the formula
    M3 = A_1_1 * (B_1_2 - B_2_2)
    */
    Dtype* M3 = strassen_make_M3_submatrix(
            A_1_1, m1, k1, k1,
            B_1_2, k1, n2, n2,
            B_2_2, k2, n2, n2);
    matrix_copyTo(M3, m1, n2, n2,
                  C_1_2, m1, n2, incRowC);
    matrix_partial_addition(C_2_2, m2, n2, incRowC,
                  M3, m2, n2, n2);
    remove_matrix(M3);
    /*
    construct M4 by the formula
    M4 = A_2_2 * (B_2_1 - B_1_1)
    */
    Dtype* M4 = strassen_make_M4_submatrix(
            A_2_2, m2, k2, k2,
            B_2_1, k2, n1, n1,
            B_1_1, k1, n1, n1);
    matrix_partial_subtraction(C_1_1, m1, n1, incRowC,
                  M4, m2, n1, n1);
    matrix_partial_subtraction(C_2_1, m2, n1, incRowC,
                  M4, m2, n1, n1);
    remove_matrix(M4);    
    /*
    construct M5 by the formula
    M5 = (A_1_1 + A_1_2) * B_2_2
    */
    Dtype* M5 = strassen_make_M5_submatrix(
            A_1_1, m1, k1, k1,
            A_1_2, m1, k2, k2,
            B_2_2, k2, n2, n2);
    matrix_partial_subtraction(C_1_1, m1, n1, incRowC,
                  M5, m1, n2, n2);
    matrix_partial_addition(C_1_2, m1, n2, incRowC,
                  M5, m1, n2, n2);
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
    matrix_partial_subtraction(C_2_2, m2, n2, incRowC,
                  M6, m2, n2, n2);
    remove_matrix(M6); 


    return;
}