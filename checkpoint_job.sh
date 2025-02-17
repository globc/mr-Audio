#!/bin/bash -l
#SBATCH --job-name=mr_audio_debug
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --output=logs/job%j%x.out
#SBATCH --error=logs/job%j_%x.err
#SBATCH --mail-user=minatocss@gmail.com
#SBATCH --mail-type=END,FAIL                     # Options: BEGIN, END, FAIL, or ALL

# Unset SLURM_EXPORT_ENV to prevent automatic export of environment variables
unset SLURM_EXPORT_ENV

# Set OMP_NUM_THREADS based on SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load required modules for the job (uncomment and modify if necessary)
module load gcc/11 cuda/12.1.1 cudnn/8.9.6.50-12.x

# Activate Conda environment for your job
source /home/hpc/g102ea/g102ea24/.conda/envs/mrAudioConda/lib/python3.9/venv/scripts/common/activate

# Set proxy environment variables if required (remove if not needed)
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Run the training script
./run_scripts/mr_BLIP/train/charades.sh

# Optionally deactivate conda at the end (if desired)
# conda deactivate
