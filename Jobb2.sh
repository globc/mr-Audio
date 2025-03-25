#!/bin/bash
#SBATCH -J mr_audio
# Please check paths (directories have to exist beforehand):
#SBATCH -e test.err
#SBATCH -o test.out
#
# CPU specification
#SBATCH -n 1 # 1 process
#SBATCH -c 4 # 4 CPU cores per process
# can be referenced as $SLURM_CPUS_PER_TASK?~@~K in the "payload" part
#SBATCH --mem-per-cpu=17500 # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 01:30:00 # in hours:minutes, or '#SBATCH -t 10' - just minutes

#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=h.maraqten@ŋmail.com # your email

#SBATCH -A kurs00079
#SBATCH -p kurs00079
#SBATCH --reservation=kurs00079
# GPU specification
#SBATCH --gres=gpu:v100:4 # 1 GPUs of type NVidia "Volta 100"
# can be referenced down below as $SLURM_GPUS_ON_NODE
# -------------------------------
# your job's "payload" in form of commands to execute, eg.
# specification from OMP_NUM_THREADS depends on your program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for checking whether and which GPUs have been allocated
# (output appears in the "#SBATCH -e" file specified above):
nvidia-smi 1>&2
# if your program supports this way of getting told how many GPUs to use:
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

ml gcc/11 python/3.8 cuda/11.8
source mraudio/bin/activate

#pip install git+https://github.com/salesforce/LAVIS --no-deps
#pip install -r requirements_xinstructblip.txt
####./scripts/X-InstructBLIP/qvh.sh
####./scripts/X-InstructBLIP/charades_sta.sh

./work/scratch/kurse/kurs00079/hm66ryjy/mr-Audio/run_scripts/mr_BLIP/train/qvh.sh
deactivate

EXITCODE=$?
# any cleanup and copy commands:
# end this job script with precisely the exit status of your scientific program above:
exit $EXITCODE
