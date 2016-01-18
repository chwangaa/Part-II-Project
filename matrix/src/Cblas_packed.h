/*! \file gemm.h
    \brief none-assembly version of GEMM algorithm for FIXED-POINT datatype
*/
#ifndef PACKED_CBLAS_GEMM_H
#define PACKED_CBLAS_GEMM_H
#include "../setting.h"
#include "blis.h"

static Dtype _C[MR*NR];

static void
dgeaxpy(int           m,
        int           n,
        double        alpha,
        const Dtype  *X,
        int           incRowX,
        int           incColX,
        Dtype        *Y,
        int           incRowY,
        int           incColY)
{
    int i, j;


    if (alpha!=1.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
static void
dgescal(int     m,
        int     n,
        Dtype  alpha,
        Dtype  *X,
        int     incRowX,
        int     incColX)
{
    int i, j;

    if (alpha!=0.0) {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] *= alpha;
            }
        }
    } else {
        for (j=0; j<n; ++j) {
            for (i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0.0;
            }
        }
    }
}


static void
dgemm_macro_kernel(int     mc,
                   int     nc,
                   int     kc,
                   double  beta,
                   Dtype  *C,
                   Dtype  *A,
                   Dtype  *B,
                   int     incRowC,
                   int     incColC)
{
    int mp = (mc+MR-1) / MR;
    int np = (nc+NR-1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;
    float alpha = 1;
    auxinfo_t* auxinfo_data = (auxinfo_t*)malloc(sizeof(auxinfo_t));
    assert(auxinfo_data);
    for (j=0; j<np; ++j) {
        nr    = (j!=np-1 || _nr==0) ? NR : _nr;

        for (i=0; i<mp; ++i) {
            mr    = (i!=mp-1 || _mr==0) ? MR : _mr;
            assert(mr == MR && nr == NR);
            if (mr==MR && nr==NR) {
                float* alpha_ptr = &alpha;
                float non_zero = 1;
                float* beta_ptr;
                if(beta != 0){
                    beta_ptr = &non_zero;
                }
                else{
                    beta_ptr = &beta;
                }
                assert(auxinfo_data);
                // fprintf(stderr, "value of A is: %p", A);
                bli_sgemm_asm_8x8(
                    kc, 
                    alpha_ptr,
                    &A[i*kc*MR],
                    &B[j*kc*NR],
                    beta_ptr,
                    &C[i*MR*incRowC+j*NR*incColC], 
                    incRowC, 
                    incColC,
                    auxinfo_data);
                // fprintf(stderr, "after: \n");
                // for(int i = 0; i < 16; i++){
                //     for(int j = 0; j < 16; j++){
                //     fprintf(stderr, "%d ", (int)A[i*16+j]);
                // }
                // fprintf(stderr, "\n");
            // }
            // fprintf(stderr, "\n");

            }
        }
    }
    free(auxinfo_data);
}

//
//  Compute C <- beta*C + alpha*A*B
//
void
packed_cblas_gemm(int            m,
         int            n,
         int            k,
         // double         alpha,
         const Dtype   *A,
         int            incRowA,
         // int            incColA,
         const Dtype   *B,
         int            incRowB,
         // int            incColB,
         // double         beta,
         Dtype         *C,
         int            incRowC
         // int            incColC
         )
{
    int mb = (m+MC-1) / MC;
    int nb = (n+NC-1) / NC;
    int kb = (k+KC-1) / KC;
    // fprintf(stderr, "kb is %d, k is %d \n ", kb, k);
    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;
    int mc, nc, kc;
    int i, j, l;

    double _beta;

    // if (alpha==0.0 || k==0) {
    //     dgescal(m, n, beta, C, incRowC, incColC);
    //     return;
    // }

    for (j=0; j<nb; ++j) {
        nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (l=0; l<kb; ++l) {
            kc    = (l!=kb-1 || _kc==0) ? KC   : _kc;
            _beta = (l==0) ? 0 : 1.0;
            // pack_B(kc, nc,
            //        &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
            //        _B);

            for (i=0; i<mb; ++i) {
                mc = (i!=mb-1 || _mc==0) ? MC : _mc;
                // pack_A(mc, kc,
                //        &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                //        _A);

                // dgemm_macro_kernel(mc, nc, kc, alpha, _beta,
                //                    &C[i*MC*incRowC+j*NC*incColC],
                //                    incRowC, incColC);
                // fprintf(stderr, "values are: %d %d %d \n", i, j, l);
                // fprintf(stderr, "kc, %d, mc: %d, KC: %d, MC: %d, l: %d", kc, mc, KC, MC, l);
                dgemm_macro_kernel(mc, nc, kc, _beta,
                                   &C[i*MC*incRowC+j*NC],
                                   &A[l*KC*m+kc*mc*i],
                                   &B[l*KC*incRowB+j*NC],
                                   incRowC, 1);
            }
            // fprintf(stderr, "value: %d beta: %d \n", (int)C[15*16], (int)_beta);
        }
    }
}


#endif