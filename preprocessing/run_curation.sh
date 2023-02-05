#!/user/bin/env bash
for nq in 20 30 40
do
	for nn in 0 5 10
	do
		python curation.py \
		-i ../data/raw/drh_20221019.csv \
		-om ../data/clean/ \
		-or ../data/reference/ \
		-nq $nq \
		-nn $nn
	done
done
