#!/bin/bash -l
#SBATCH --job-name=mr_audio_charades_AdioOnly_BEATS_wo_QF
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:2 -C a100_80

#SBATCH --output=logs/job_%j_%x.out
#SBATCH --error=logs/job_%j_%x.err
#SBATCH --mail-user=h.maraqten@gmail.com
#SBATCH --mail-type=END,FAIL                     # Options: BEGIN, END, FAIL, or ALL


unset SLURM_EXPORT_ENV

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

#ml gcc/11 python/3.8 cuda/11.8
#ml gcc/11 python/3.9-anaconda cuda/12.1

#source $WORK/venvs/mrBlipAudio/bin/activate
conda init
conda activate mrAudioConda

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

ml gcc/11 cuda/12.1.1 cudnn/8.9.6.50-12.x
#conda activate mrBlipAudio

./run_scripts/mr_BLIP/train/charades.sh
#conda deactivate

