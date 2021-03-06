#!/bin/bash
#PBS -l walltime=00:08:00,nodes=1:ppn=4
#PBS -N merge_sort_openmp
#PBS -q batch

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=$PBS_NUM_PPN

for thr_num in 1 2 3 4 5 6 7 8; do \
    ./main $thr_num; \
done
