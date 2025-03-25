#!/bin/bash -l
#
#SBATCH -e test.err
#SBATCH -o test.out

#SBATCH -n 1 # 1 process
#SBATCH -c 4 # 4 CPU cores per process

#SBATCH --time=24:00:00

#SBATCH --gres=gpu:a100:4 -C a100_80
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HPC_SCRATCH=$WORK

ml gcc/11 cuda
source ~/miniconda3/bin/activate
conda activate mraudio
pip install -r requirements.txt

./run_scripts/mr_Audio/train/X-InstructBLIP/charades_sta.sh
conda deactivate


