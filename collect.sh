#!/bin/bash
for ((COUNTER=64; COUNTER<=2048; COUNTER+=64));
do
	for i in {1}
	do
	matrix/test/nopadding $COUNTER $COUNTER $COUNTER >> data/hybrid_32.csv 
	done
done