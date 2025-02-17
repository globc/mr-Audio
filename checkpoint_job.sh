#!/bin/bash -l
#SBATCH --job-name=mr_audio_debug
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --mem=32G
#SBATCH --output=logs/job%j%x.out
#SBATCH --error=logs/job%j_%x.err
#SBATCH --mail-user=minatocss@gmail.com
#SBATCH --mail-type=END,FAIL

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=8  # Adjust this based on your CPU cores
export OMP_STACKSIZE=512M  # Increase/decrease if necessary
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

conda init bash
conda activate mrAudioConda

ml gcc/11 cuda/12.1.1 cudnn/8.9.6.50-12.x

./run_scripts/mr_BLIP/train/charades.sh
