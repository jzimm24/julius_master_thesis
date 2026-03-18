#!/bin/bash -l
#SBATCH -N 1 -p intelsr_devel --exclusive --time=00:05:00
#SBATCH --mail-type=ALL   # notifications for job done & fail
#SBATCH --mail-user=s64jzimm@uni-bonn.de #user email for updates

echo Hello World!

# Activate your venv
source /home/s64jzimm_hpc/julius_master_thesis/.venv1/bin/activate

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# Run Python script
echo ""
echo "***** LAUNCHING *****"
echo `date '+%F %H:%M:%S'`
echo ""

srun python src/CNN_test.py

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""

exit 0