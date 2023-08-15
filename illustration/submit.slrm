#!/bin/bash
#SBATCH -p batch
#SBATCH -t 5-00:00:00
#SBATCH -N 1       
#SBATCH -n 1     
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --output=code_%A.out
#SBATCH--mail-type=FAIL
#SBATCH--mail-type=END
#SBATCH--mail-user=zhiqiang.liao@aalto.fi

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Create runtime environment
cd /scratch/work/liaoz1/papers/WRCNLS/simu/ill2/

module load julia

# Run Julia
srun julia -t $SLURM_CPUS_PER_TASK ill.jl
