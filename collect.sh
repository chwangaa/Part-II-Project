#!/bin/bash
for ((COUNTER=64; COUNTER<=2048; COUNTER += 64));
do
	for i in {1}
	do
	strassen/build/benchmark $COUNTER $COUNTER $COUNTER 0 &>> data/naive_blis.csv
	strassen/build/benchmark $COUNTER $COUNTER $COUNTER 1 &>> data/naive_cblas.csv
	strassen/build/benchmark $COUNTER $COUNTER $COUNTER 5 &>> data/naive.csv
	done
done
