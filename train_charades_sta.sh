#!/bin/bash -l
#
#SBATCH -e test.err
#SBATCH -o test.out

#SBATCH -n 1 # 1 process
#SBATCH -c 4 # 4 CPU cores per process

#SBATCH --time=24:00:00

#SBATCH --gres=gpu:a100:8 -C a100_80
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export WANDB_API_KEY=f851d49cb27f7d3aefb3ad148212737266995e7e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HPC_SCRATCH=$WORK

ml gcc/11 cuda
source ~/miniconda3/bin/activate
conda activate mraudio
# pip install git+https://github.com/salesforce/LAVIS --no-deps
pip install -r requirements.txt

./run_scripts/mr_BLIP/train/charades.sh
conda deactivate