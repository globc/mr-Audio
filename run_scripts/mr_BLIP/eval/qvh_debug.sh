# Should return:
# {'agg_metrics': 57.55899999999999, 'r1': {'0.5': 76.16, '0.55': 72.1, '0.6': 69.2, '0.65': 66.24, '0.7': 62.63, '0.75': 59.73, '0.8': 54.64, '0.85': 49.29, '0.9': 38.92, '0.95': 26.68}, 'mAP': {'0.5': 68.5, '0.55': 65.19, '0.6': 62.91, '0.65': 60.43, '0.7': 57.48, '0.75': 55.06, '0.8': 50.79, '0.85': 45.96, '0.9': 36.4, '0.95': 24.94, 'average': 52.77}, 'mIoU': 0.703218087517246, 'invalid_predictions': 0.014175257731958763, 'total': 1552}

CUDA_VISIBLE_DEVICES="" python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/mr_BLIP/eval/qvh.yaml