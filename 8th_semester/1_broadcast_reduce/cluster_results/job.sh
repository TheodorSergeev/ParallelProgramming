#!/bin/bash
#PBS -l walltime=00:08:00,nodes=7:ppn=4
#PBS -N example_job
#PBS -q batch


cd $PBS_O_WORKDIR

rm reduce_bcast_speed.txt -f ; \
for p_num in 2 4 6 8 10 12 14 16; do \
    echo "p = $p_num"; \ 
    mpirun --hostfile $PBS_NODEFILE -np $p_num ./main; \
done
