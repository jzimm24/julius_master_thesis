#! bin/bash
# This is a comment in a SLURM script
# #SBATCH is a directive for the SLURM scheduler.


#SBATCH --partition=aifa-science  # for the main queue
# or
# #SBATCH --partition=science-old  # alternate old
#SBATCH --job-name=mpi_matplotlib_test
#SBATCH -N 1        # number of nodes
#SBATCH -n 4        # number of cores (change this to match your needs)
#SBATCH -t 01:00:00    # Maximum requested time (hrs:min:sec)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL   # notifications for job done & fail
#SBATCH --mail-user=adev@astro.uni-bonn.de #user email for updates

# If you have a specific Python environment, activate it here
# source activate myenv

# Or for Anaconda
# Intialize Conda, need the Conda shell file loc
source /vol/aibn49/data1/adev/opt/anaconda3/etc/profile.d/conda.sh
# Activate your environment
conda activate ccat-prime

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"

# In case you are using libraries from a different location specified
# in your library path, then you need to add them here.
# export LD_LIBRARY_PATH=<...>:$LD_LIBRARY_PATH

echo ""
echo "***** LAUNCHING *****"
echo `date '+%F %H:%M:%S'`
echo ""

# Run python code, the program with flags
mpirun -np $SLURM_NTASKS python3 src/CNN_test.py

# If Without mpi
# python3 matplotlib_test.py

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""

exit 0
