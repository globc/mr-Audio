#!/bin/bash
#SBATCH -J mr_audio
#SBATCH --output=logs/job%j%x.out
#SBATCH --error=logs/job%j%x.err

# CPU specification
#SBATCH -n 1                # 1 process
#SBATCH -c 4                # 4 CPU cores per process
#SBATCH -t 24:00:00         # Time limit in hours:minutes

# Notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h.maraqten@gmail.com

# Cluster resources
#SBATCH -A kurs00079
#SBATCH -p kurs00079
#SBATCH --reservation=kurs00079

# GPU specification
#SBATCH --gres=gpu:a100:4   # 4 GPUs of type A100

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi 1>&2
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

ml gcc/11 python/3.8 cuda/11.8
source mrAudio_venv/bin/activate

# Your commands
./run_scripts/mr_BLIP/train/qvh.sh

# Capture and return exit code
EXITCODE=$?
exit $EXITCODE
