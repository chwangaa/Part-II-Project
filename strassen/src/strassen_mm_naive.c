#include "strassen_util.h"

void strassen_mm(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

	// the matrices must have positive dimension
	debug_assert(m > 0);
	debug_assert(n > 0);
	debug_assert(k > 0);
	
	/* check if the base case has reached
	   here we recurse until all the dimension are smaller than 2
	*/
	if(m <= limit_X && n <= limit_X && k <= limit_X){
        return matrix_multiplication_base_case(
            m, n, k,
            A, incRowA,
            B, incRowB,
            C, incRowC);
	}


	/*
	We divide A, B, C into subsections as the following:
	A = |A_1_1, A_1_2|   B = |B_1_1, B_1_2|   C = |C_1_1, C_1_2|
	    |A_2_1, A_2_2|       |B_2_1, B_2_2|       |C_2_1, C_2_2|


	We first calculate 7 temporary matrices as the follow:
	M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
	M2 = (A_2_1 + A_2_2) * B_1_1
	M3 = A_1_1 * (B_1_2 - B_2_2)
	M4 = A_2_2 * (B_2_1 - B_1_1)
	M5 = (A_1_1 + A_1_2) * B_2_2
	M6 = (A_2_1 - A_1_1) * (B_1_1 + B_1_2)
	M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)

	Then, we compute C section by section according to:
	C_1_1 = M1 + M4 - M5 + M7
	C_1_2 = M3 + M5
	C_2_1 = M2 + M4
	C_2_2 = M1 - M2 + M3 + M6 
	*/
	const unsigned int m1 = m / 2;
	const unsigned int m2 = m - m1;	
	const unsigned int n1 = n / 2;
	const unsigned int n2 = n - n1;
	const unsigned int k1 = k / 2;
	const unsigned int k2 = k - k1;
	
	const Dtype* A_1_1 = A;
	const Dtype* A_1_2 = A_1_1 + k1;
	const Dtype* A_2_1 = A_1_1 + incRowA*m1;
	const Dtype* A_2_2 = A_2_1 + k1;

	Dtype* C_1_1 = C;
	Dtype* C_1_2 = C_1_1 + n1;
	Dtype* C_2_1 = C_1_1 + incRowC*m1;
	Dtype* C_2_2 = C_2_1 + n1;

	const Dtype* B_1_1 = B;
	const Dtype* B_1_2 = B_1_1 + n1;
	const Dtype* B_2_1 = B_1_1 + incRowB*k1;
	const Dtype* B_2_2 = B_2_1 + n1;

	// first construct the temporary for A_1_1 + A_2_2
	// this version assumes square matrices, m1 == n1 == k1
	debug_assert(m1 == k1);
	debug_assert(m1 == n1);
	debug_assert(m1 == m2);
	/*
	construct M1 by the formula
	M1 = (A_1_1 + A_2_2) * (B_1_1 + B_2_2)
	*/
	
	// T1 = (A_1_1 + A_2_2)
	Dtype* T1 = make_matrix(m1, k1);
	matrix_addition(m1, k1,
		A_1_1, incRowA,
		A_2_2, incRowA,
		T1, k1);
   
    // T2 = (B_1_1 + B_2_2)
    Dtype* T2 = make_matrix(m1, k1);
    matrix_addition(m1, k1,
    	B_1_1, incRowB,
    	B_2_2, incRowB,
    	T2, k1);

    // M1 = T1 * T2
    Dtype* M1 = make_matrix(m1, k1);
    strassen_mm(
    	m1, n1, k1,
    	T1, k1,
    	T2, k1,
    	M1, k1);

    // now that T1, T2 can hold new temporaries
    /*
	construct M2 by the formula
	M2 = (A_2_1 + A_2_2) * B_1_1
    */
    // T1 = A_2_1 + A_2_2
    matrix_addition(m1, k1,
    	A_2_1, incRowA,
    	A_2_2, incRowA,
    	T1, k1);

    Dtype* M2 = make_matrix(m1, k1);
    strassen_mm(
    	m1, n1, k1,
    	T1, k1,
    	B_1_1, incRowB,
    	M2, k1);

    /*
	construct M3 by the formula
	M3 = A_1_1 * (B_1_2 - B_2_2)
    */
    // T1 = B_1_2 - B_2_2
    matrix_subtraction(m1, k1,
    	B_1_2, incRowB,
    	B_2_2, incRowB,
    	T1, k1);

    Dtype* M3 = make_matrix(m1, k1);
    strassen_mm(
    	m1, n1, k1,
    	A_1_1, k1,
    	T1, k1,
    	M3, k1);

    /*
	construct M4 by the formula
	M4 = A_2_2 * (B_2_1 - B_1_1)
    */
    // T1 = B_2_1 - B_1_1
    matrix_subtraction(m1, k1,
    	B_2_1, incRowB,
    	B_1_1, incRowB,
    	T1, k1);

    Dtype* M4 = make_matrix(m1, k1);
    strassen_mm(
    	m1, n1, k1,
    	A_2_2, k1,
    	T1, k1,
    	M4, k1);


    /*
	construct M5 by the formula
	M5 = (A_1_1 + A_1_2) * B_2_2
    */
    // T1 = A_1_1 + A_1_2
    matrix_addition(m1, k1,
    	A_1_1, incRowA,
    	A_1_2, incRowA,
    	T1, k1);

    Dtype* M5 = make_matrix(m1, k1);
    strassen_mm(
    	m1, n1, k1,
    	T1, k1,
    	B_2_2, incRowB,
    	M5, k1);


	/*
	construct M6 by the formula
	M6 = (A_2_1 - A_1_1) * (B_1_1 + B_1_2)
	*/
	
	// T1 = (A_2_1 - A_1_1)
	matrix_subtraction(m1, k1,
		A_2_1, incRowA,
		A_1_1, incRowA,
		T1, k1);
   
    // T2 = (B_1_1 + B_1_2)
    matrix_addition(m1, k1,
    	B_1_1, incRowB,
    	B_1_2, incRowB,
    	T2, k1);

    // M6 = T1 * T2
    Dtype* M6 = make_matrix(m1, k1);
    strassen_mm(
    	m1, n1, k1,
    	T1, k1,
    	T2, k1,
    	M6, k1);

	/*
	construct M7 by the formula
	M7 = (A_1_2 - A_2_2) * (B_2_1 + B_2_2)
	*/
	
	// T1 = (A_1_2 - A_2_2)
	matrix_subtraction(m1, k1,
		A_1_2, incRowA,
		A_2_2, incRowA,
		T1, k1);
   
    // T2 = (B_2_1 + B_2_2)
    matrix_addition(m1, k1,
    	B_2_1, incRowB,
    	B_2_2, incRowB,
    	T2, k1);

    // M7 = T1 * T2
    Dtype* M7 = make_matrix(m1, k1);
    strassen_mm(
    	m1, n1, k1,
    	T1, k1,
    	T2, k1,
    	M7, k1);

    /*
    compute C_1_1 by the formula
    C_1_1 = M1 + M4 - M5 + M7
    */
    // C_1_1 = M1 + M4
    matrix_addition(m1, k1,
    	M1, k1,
    	M4, k1,
    	C_1_1, incRowC);
    // C_1_1 += M7
    matrix_partial_addition(C_1_1, m1, k1, incRowC,
                            M7, m1, k1, k1);
    // C_1_1 -= M5
    matrix_partial_subtraction(C_1_1, m1, k1, incRowC,
                            M5, m1, k1, k1);
    /*
    compute C_1_2 by the formula
    C_1_2 = M3 + M5
    */
    matrix_addition(m1, k1,
    	M3, k1,
    	M5, k1,
    	C_1_2, incRowC);

    /*
    compute C_2_1 by the formula
    C_2_1 = M2 + M4
    */
    matrix_addition(m1, k1,
    	M2, k1,
    	M4, k1,
    	C_2_1, incRowC);

    /*
    compute C_2_2 by the formula
    C_2_2 = M1 - M2 + M3 + M6
    */
    // C_2_2 = M1 - M2
    matrix_subtraction(m1, k1,
    	M1, k1,
    	M2, k1,
    	C_2_2, incRowC);
    
    // C_2_2 += M3
    matrix_partial_addition(C_2_2, m1, k1, incRowC,
                            M3, m1, k1, k1);
    // C_2_2 += M6
    matrix_partial_addition(C_2_2, m1, k1, incRowC,
                            M6, m1, k1, k1);

    /*
    remove the working space
    */
    remove_matrix(T1);
    remove_matrix(T2);
    remove_matrix(M1);
    remove_matrix(M2);
    remove_matrix(M3);
    remove_matrix(M4);
    remove_matrix(M5);
    remove_matrix(M6);
    remove_matrix(M7);

}