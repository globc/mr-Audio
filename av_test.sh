#!/bin/bash -l
#SBATCH --job-name=video_shape_extraction      # Job name
#SBATCH --output=logs/job_%j.out              # Standard output log
#SBATCH --error=logs/job_%j.err               # Standard error log
#SBATCH --time=06:30:00                       # Max runtime (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # Number of CPU cores
#SBATCH --gres=gpu:a40:1                      # GPU resource allocation

# Load necessary modules
module load gcc/11 cuda/12.1.1                # Adjust versions according to the cluster environment

# Activate Python virtual environment (if needed)
source $WORK/venvs/mrBlipAudio/bin/activate

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Run the Python program
python extract_av_test.py --input_dir /home/atuin/g102ea/shared/videos/Charades_v1 --output_csv video_shapes.csv


