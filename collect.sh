#!/bin/bash
for ((COUNTER=2; COUNTER<=16384; COUNTER*=2));
do
	for i in {1..3}
	do
	./order $COUNTER $COUNTER $COUNTER >> data/strassen_gemm_order_256.csv 
	# ./cache_oblivious $COUNTER $COUNTER $COUNTER >> cache_oblivious_O3_result.csv
	# ./cache_oblivious_no_optimization $COUNTER $COUNTER $COUNTER >> cache_oblivious_result.csv
	# ./strassen_no_optimization $COUNTER $COUNTER $COUNTER >> strassen_result.csv
	# ./simple $COUNTER $COUNTER $COUNTER >> simple_O3_result.csv
	# ./simple_no_optimization $COUNTER $COUNTER $COUNTER >> simple_result.csv
	done
done