#ifndef CACHE_OBLIVIOUS_MATRIX_MULTIPLICATION_H
#define CACHE_OBLIVIOUS_MATRIX_MULTIPLICATION_H
#include "setting.h"
#include "SimpleMatrixMultiplication.h"
#include <stdio.h>

void cache_oblivious_matrix_multiplication(
    const unsigned int m,
    const unsigned int n,
    const unsigned int k,
    const Dtype *A, const int incRowA,
    const Dtype *B, const int incRowB,
    Dtype *C, const int incRowC){

	// the matrices must have positive dimension
	assert(m > 0);
	assert(n > 0);
	assert(k > 0);
	
	/* check if the base case has reached
	   here we recurse until all the dimension are smaller than 2
	*/
	if(m <= 2 && n <= 2 && k <= 2){
		return SimpleMatrixMultiplication(
			m, n, k,
			A, incRowA,
			B, incRowB,
			C, incRowC);
	}



	/* recursion cases*/

	/*	Case 1, when m >= max(n, k)
		we split it to:
		C = |A_1_1| * B
		    |A_2_1|
	 */
	if(m >= n && m >= k){

		const unsigned int m1 = m / 2;
		const unsigned int m2 = m - m1;	
		const Dtype* A_1_1 = A;
		const Dtype* A_2_1 = A_1_1 + incRowA*m1;

		Dtype* C_1_1 = C;
		Dtype* C_2_1 = C_1_1 + incRowC*m1;

		cache_oblivious_matrix_multiplication(
			m1, n, k, 
			A_1_1, incRowA,
			B, incRowB,
			C_1_1, incRowC);

		return cache_oblivious_matrix_multiplication(
			m2, n, k,
			A_2_1, incRowA,
			B, incRowB,
			C_2_1, incRowC);
		}


	/*	Case 2, when n >= max(m, k)
		we split it to:
		C = A * |B_1_1, B_1_2|
	 */

	if(n >= m && n >= k){

		const unsigned int n1 = n / 2;
		const unsigned int n2 = n - n1;
		const Dtype* B_1_1 = B;
		const Dtype* B_1_2 = B_1_1 + n1;
		Dtype* C_1_1 = C;
		Dtype* C_1_2 = C_1_1 + n1;

		cache_oblivious_matrix_multiplication(
			m, n1, k, 
			A, incRowA,
			B_1_1, incRowB,
			C_1_1, incRowC);

		return cache_oblivious_matrix_multiplication(
			m, n2, k,
			A, incRowA,
			B_1_2, incRowB,
			C_1_2, incRowC);
	}

	/*	Case 3, when k >= max(m, n)
		we split it to:
		C = |A_1_1, A_1_2| *|B_1_1|
		                    |B_2_1|
	 */	
	const unsigned int k1 = k / 2;
	const unsigned int k2 = k - k1;
	const Dtype* A_1_1 = A;
	const Dtype* A_1_2 = A_1_1 + k1;
	const Dtype* B_1_1 = B;
	const Dtype* B_2_1 = B_1_1 + incRowB*k1;

	cache_oblivious_matrix_multiplication(
			m, n, k1, 
			A_1_1, incRowA,
			B_1_1, incRowB,
			C, incRowC);

	return cache_oblivious_matrix_multiplication(
			m, n, k2,
			A_1_2, incRowA,
			B_2_1, incRowB,
			C, incRowC);
    	}


#endif