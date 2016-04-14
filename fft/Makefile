FFT_INC = -I$(HOME)/Documents/Project/fft/include
LIB = -lm

TEST_CC = gcc -std=c99 $(FFT_INC) -O3 $^ -o $@ -lfftw3f -lm

all: fft.o

test: test1d test2d

test1d: test_fft test_ifft
	./test_fft
	./test_ifft

test2d: test_fft2 test_ifft2 test_conv test_shift_ifft test_linear_correlation test_linear_correlation3d
	./test_fft2
	./test_ifft2
	./test_conv
	./test_shift_ifft
	./test_linear_correlation
	./test_linear_correlation3d

fft2.o: src/fft2.c include/fft_setting.h
	gcc -std=c99 $(FFT_INC) -O3 -c -o $@ $< -lm

fft.o: src/fft.c include/fft_setting.h
	gcc -std=c99 $(FFT_INC) -O3 -c -o $@ $< -lfftw3 -lm

fftw.o: src/fft2_fftw3.c include/fft_setting.h
	gcc -std=c99 $(FFT_INC) -O3 -c -o $@ $< -lm

fftw_test: fftw.o test/fftw_test.c
	$(TEST_CC)

test_ifft: fft.o test/ifft_test.c
	$(TEST_CC)

test_fft: fft.o test/fft_test.c
	$(TEST_CC)

test_fft2: fft2.o test/fft2_test.c
	$(TEST_CC)

test_ifft2: fft2.o test/ifft2_test.c
	$(TEST_CC)

test_conv: fft2.o test/conv_test.c
	$(TEST_CC)

test_shift_ifft: fft2.o test/shift_ifft2_test.c
	$(TEST_CC)

test_linear_correlation: fft2.o test/linear_correlation_test.c
	$(TEST_CC)

test_linear_correlation3d: fft2.o test/linear_correlation3_test.c
	$(TEST_CC)
