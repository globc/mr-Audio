#!/bin/bash -l
#SBATCH --job-name=mr_audio_charades
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:4

#SBATCH --output=logs/job_%j_%x.out
#SBATCH --error=logs/job_%j_%x.err

unset SLURM_EXPORT_ENV

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

#ml gcc/11 python/3.8 cuda/11.8
#ml gcc/11 python/3.9-anaconda cuda/12.1
#source $WORK/venvs/mrAudio/bin/activate

ml gcc/11 cuda/12.1.1
conda activate mrCLAP

./run_scripts/mr_BLIP/train/charades.sh
conda deactivate