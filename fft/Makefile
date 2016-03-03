FFT_INC = -I$(HOME)/Cambridge/Part-II-Project/fft/include
LIB = -lm

TEST_CC = gcc -std=c99 $(FFT_INC) -O3 $^ -o $@ -lm

all: fft.o

test: test_fft test_ifft test_conv test_shift_ifft test_linear_correlation
	./test_fft
	./test_ifft
	./test_conv
	./test_shift_ifft
	./test_linear_correlation

fft.o: src/fft2.c include/fft_setting.h
	gcc -std=c99 $(FFT_INC) -O3 -c -o $@ $< -lm

test_fft: fft.o test/fft_test.c
	$(TEST_CC)

test_ifft: fft.o test/ifft_test.c
	$(TEST_CC)

test_conv: fft.o test/conv_test.c
	$(TEST_CC)

test_shift_ifft: fft.o test/shift_ifft_test.c
	$(TEST_CC)

test_linear_correlation: fft.o test/linear_correlation_test.c
	$(TEST_CC)

test_linear_correlation3d: fft.o test/linear_correlation3_test.c
	$(TEST_CC)
