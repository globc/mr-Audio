export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
echo $CUDA_VISIBLE_DEVICES

export MASTER_PORT=29511
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo $NUM_GPUS


python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT evaluate.py --cfg-path lavis/projects/mr_Audio/eval/xinstructblip/qvh.yaml