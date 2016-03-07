/*! \file gemm.h
    \brief none-assembly version of GEMM algorithm for FIXED-POINT datatype
*/
#include "strassen_util.h"
#include "blis.h"

#include <emmintrin.h>
#include <immintrin.h>

static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   Dtype  *C,
                   Dtype  *A,
                   Dtype  *B,
                   int     incRowC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;
    float alpha = 1;

    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;
        Dtype* restrict B_r = &B[j*kc*NR];

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            assert(mr == MR && nr == NR);
            if (mr==MR && nr==NR) {
                float* alpha_ptr = &alpha;
                float non_zero = 1;
                float beta = 0;
                float* beta_ptr = &beta;

                Dtype* restrict A_r = &A[i*kc*MR];
                Dtype* restrict C_r = &C[i*MR*incRowC+j*NR];
                assert(((unsigned long)C_r & 31) == 0);
                assert(((unsigned long)A_r & 31) == 0);
                assert(((unsigned long)B_r & 31) == 0);
                assert((incRowC % 8 == 0));
                bli_sgemm_asm_8x8(
                    kc, 
                    alpha_ptr,
                    A_r,
                    B_r,
                    beta_ptr,
                    C_r, 
                    incRowC, 
                    1,
                    0);

            }
        }
    }
}

void
packed_mm(int            m,
         int            n,
         int            k,
         const Dtype   *A,
         int            incRowA,
         const Dtype   *B,
         int            incRowB,
         Dtype         *C,
         int            incRowC
         )
{
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;
    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;
    int mc, nc, kc;
    int i, j, l;



    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;


            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                dgemm_macro_kernel(mc, nc, kc,
                                   &C[i*MC*incRowC+j*NC],
                                   &A[i*MC*incRowA+l*KC],
                                   &B[l*KC*incRowB+j*NC],
                                   incRowC);
            }
        }
    }
}