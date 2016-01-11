# Optimisations to apply to standard compilations.
OPTIMISATIONS=-O3

# Libraries required.
LIBRARIES=-lm -lcblas

DEPENDENCIES = 

BLIS_PREFIX = $(HOME)/blis
BLIS_LIB = $(BLIS_PREFIX)/lib/libblis.a
BLIS_INC = $(BLIS_PREFIX)/include/blis
all: nopadding padsquare power base sgemm minimalspace

strassen-base.o: src/StrassenMatrixMultiplication.c
	gcc -std=c99 -O3 -c -o strassen-base.o src/StrassenMatrixMultiplication.c -lcblas

strassen-nopadding.o: src/StrassenMatrixMultiplication_nopadding2.c setting.h
	gcc -std=c99 -O3 -c -o strassen-nopadding.o src/StrassenMatrixMultiplication_nopadding2.c -lcblas

strassen-padtopowermatrix.o: src/StrassenMatrixMultiplication_refactored.c
	gcc -std=c99 -O3 -c -o strassen-padtopowermatrix.o src/StrassenMatrixMultiplication_refactored.c -lcblas

strassen-padtosquarematrix.o: src/StrassenMatrixMultiplication_refactored2.c
	gcc -std=c99 -O3 -c -o strassen-padtosquarematrix.o src/StrassenMatrixMultiplication_refactored2.c -lcblas

strassen_test.o: strassen_test.c setting.h
	gcc -std=c99 -O3 -c -o strassen_test.o strassen_test.c

strassen-minimalspace.o: src/StrassenMatrixMultiplication_minimalspace.c
	gcc -std=c99 -I$(BLIS_INC) -O3 -c -o strassen-minimalspace.o src/StrassenMatrixMultiplication_minimalspace.c

base:
	gcc -g -std=c99 -O3 -o test/base_addition baseline_test.c

padsquare: strassen_test.o strassen-padtosquarematrix.o
	gcc -g -std=c99 -O3 -o test/padsquare strassen_test.o strassen-padtosquarematrix.o -lcblas	

power: strassen_test.o strassen-padtopowermatrix.o
	gcc -g -std=c99 -O3 -o test/padpower strassen_test.o strassen-padtopowermatrix.o -lcblas

nopadding: strassen_test.o strassen-nopadding.o
	gcc -g -std=c99 -O3 -o test/nopadding strassen_test.o strassen-nopadding.o -lcblas

minimalspace: strassen_test.o strassen-minimalspace.o
	gcc -std=c99 strassen_test.o strassen-minimalspace.o -O3 $(BLIS_LIB) -o test/minimalspace -lm -fopenmp

sgemm:
	gcc -std=c99 -O3 -o test/sgemm sgemm_test.c -lcblas -g

blis_gemm_main.o: blis_gemm_test.c
	gcc -std=c99 -O3 -g -I$(BLIS_INC) -c -o blis_gemm_test.o blis_gemm_test.c	

blis_gemm: blis_gemm_main.o
	gcc -std=c99 blis_gemm_test.o -g -O3 $(BLIS_LIB) -o test/blis_gemm -lm -fopenmp

clean:
	rm strassen*.o
