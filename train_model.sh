#!/bin/bash -l
#SBATCH --job-name=mr_audio_charades
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4 -C a100_80

#SBATCH --output=logs/job_%j_%x.out
#SBATCH --error=logs/job_%j_%x.err


unset SLURM_EXPORT_ENV

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

#ml gcc/11 python/3.8 cuda/11.8
#ml gcc/11 python/3.9-anaconda cuda/12.1

source $WORK/venvs/mrBlipAudio/bin/activate

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

ml gcc/11 cuda/12.1.1 cudnn/8.9.6.50-12.x
#conda activate mrBlipAudio

./run_scripts/mr_Audio/blip2_mr_audio_xinstructblip/charades_sta.sh
#conda deactivate
