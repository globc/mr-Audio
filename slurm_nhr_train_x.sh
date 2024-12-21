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

export HPC_SCRATCH=$WORK

ml gcc/11 cuda/11.8
source ~/miniconda3/bin/activate
conda activate mraudio2
pip install -r requirements_xinstructblip.txt
export CFG_PATH=lavis/projects/mr_Audio/train/xinstructblip/qvh_nhr.yaml

./run_scripts/mr_Audio/train/X-InstructBLIP/qvh.sh
conda deactivate


