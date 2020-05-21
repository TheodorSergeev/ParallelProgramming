#!/bin/bash
#PBS -l walltime=00:08:00,nodes=7:ppn=4
#PBS -N heat_eq
#PBS -q batch

cd $PBS_O_WORKDIR

for p_num in 1 2 3 4 5 6 7 8; do \
    echo "p = $p_num"; \
    mpirun --hostfile $PBS_NODEFILE -np $p_num ./main; \
done
