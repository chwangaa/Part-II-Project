STRASSEN_INC = -I$(HOME)/Cambridge/Part-II-Project/strassen/include
LIB = -lm -lcblas

BLIS_PREFIX = $(HOME)/blis
BLIS_LIB = $(BLIS_PREFIX)/lib/libblis.a
BLIS_INC = $(BLIS_PREFIX)/include/blis


STRASSEN_CC = gcc -w -std=c99 $(STRASSEN_INC) -O3 -c -o $@ $< -lm -lcblas -mavx


test: test_strassen_mm
	./build/test_strassen_mm

matrix_mul_strassen.o: src/strassen_mm.c include/matrix.h include/matrix_util.h include/strassen_util.h
	$(STRASSEN_CC)

matrix_mul_base.o: src/matrix_mul_base.c include/matrix.h
	$(STRASSEN_CC)

matrix_mul_cblas.o: src/cblas_mm.c include/matrix.h
	$(STRASSEN_CC)

matrix_mul_blis.o: src/blis_mm.c include/matrix.h
	gcc -w -std=c99 $(STRASSEN_INC) -I$(BLIS_INC) -O3 -c -o matrix_mul_blis.o src/blis_mm.c


matrix_operations.o: src/basic_matrix_operations.c include/matrix.h
	$(STRASSEN_CC)


strassen: matrix_mul_strassen.o matrix_mul_base.o matrix_operations.o matrix_mul_cblas.o matrix_mul_blis.o
	ar rvs include/strassen.a matrix_mul_strassen.o matrix_mul_base.o matrix_operations.o matrix_mul_cblas.o matrix_mul_blis.o
	rm *.o

test_strassen_mm: strassen test/matrix_mul_test.c
	gcc -w -std=c99 $(STRASSEN_INC) -I$(BLIS_INC) -O3 -o build/test_strassen_mm test/matrix_mul_test.c include/strassen.a $(BLIS_LIB) -lm -lcblas -fopenmp

benchmark: strassen test/mm_performance_benchmark.c
	gcc -w -std=c99 $(STRASSEN_INC) -I$(BLIS_INC) -O3 -o build/mm_benchmark test/mm_performance_benchmark.c include/strassen.a $(BLIS_LIB) -lm -lcblas -fopenmp
	./build/mm_benchmark

benchmark_input: strassen test/mm_benchmark_given_input.c
	gcc -w -std=c99 $(STRASSEN_INC) -I$(BLIS_INC) -O3 -o build/benchmark test/mm_benchmark_given_input.c include/strassen.a $(BLIS_LIB) -lm -lcblas -fopenmp
