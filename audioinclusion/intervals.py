import torch
import torchvision.io
import decord
from decord import VideoReader, AudioReader
import torchaudio
import os
#from audioinclusion.AudioEmbeddingsCLAP import CLAPAudioEmbeddings

def load_video_frames_with_audio(video_path,
                                      n_frms=float('inf'),
                                      height=-1,
                                      width=-1,
                                      sampling="uniform",
                                      clip_proposal=None, target_sr=48000):
    """
    Load video frames and extract corresponding audio segments efficiently.

    Args:
        video_path (str): Path to the video file.
        n_frms (int, optional): Number of frames to sample. Defaults to inf (all frames).
        height (int, optional): Height to resize the video frames to. Defaults to -1 (original height).
        width (int, optional): Width to resize the video frames to. Defaults to -1 (original width).
        sampling (str, optional): Sampling strategy for frame selection.
                                One of ["uniform", "random", "headtail"]. Defaults to "uniform".
        clip_proposal (tuple, optional): Start and end time (in seconds) to extract a clip from the video.
                                    Defaults to None (entire video).

    Returns:
        tuple: A tuple containing:
            - frms (torch.Tensor): Video frames of shape (C, T, H, W), where:
                - C: Number of color channels (e.g., 3 for RGB)
                - T: Number of sampled frames
                - H: Height of the frames
                - W: Width of the frames
            - indices (list[int]): List of selected frame indices.
            - fps (float): Frames per second of the video.
            - audio_segments (list[torch.Tensor]): List of audio tensors, where each tensor has shape (C, T ,T_audio):
                - C: Number of audio channels (e.g., 1 for mono, 2 for stereo)
                - T: Number of sampled audio clips
                - T_audio: Number of audio samples in the segment.
    """

    video, audio, info = torchvision.io.read_video(video_path, pts_unit="sec")
    sample_rate = info["audio_fps"]
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sample_rate != target_sr:
        audio = resample_audio(audio, sample_rate, target_sr)
        sample_rate = target_sr

    fps = info["video_fps"]
    num_frames = video.shape[0]
    n_frms = min(n_frms, num_frames)

    # Apply clip proposal if provided
    if clip_proposal is None:
        start, end = 0, num_frames
    else:
        start, end = int(clip_proposal[0] * fps), int(clip_proposal[1] * fps)
        if start < 0:
            start = 0
        if end > num_frames:
            end = num_frames

    indices, intervals = sample_intervals_torch(sampling, start, end, n_frms, num_frames)

    # Extract video frames
    frms = torch.index_select(video, 0, indices)
    frms = thwc_to_cthw(frms)

    # Calculate segment durations efficiently
    frame_lengths = intervals[:, 1] - intervals[:, 0]
    segment_duration = int(frame_lengths.sum() / len(intervals) * (1 / fps) * sample_rate)

    # Split audio into equally long sequences
    audio_segments = torch.split(audio, segment_duration, dim=1)
    padded_segments = [pad_or_random_crop(seg, 48000) for seg in audio_segments[:n_frms]]
    

    if height > 0 and width > 0:
        frms = torch.nn.functional.interpolate(frms, size=(height, width), mode='bilinear', align_corners=False)

    return frms, indices.tolist(), fps, torch.stack(padded_segments, dim=1).squeeze(0), sample_rate
    #return frms, indices.tolist(), fps, audio.squeeze(0), sample_rate

import sys
MAX_INT=sys.maxsize
decord.bridge.set_bridge("torch")

def load_video_with_audio(video_path,
                          n_frms=float('inf'),
                          height=-1,
                          width=-1,
                          sampling="uniform",
                          clip_proposal=None,
                          target_sr=48000):
    """
    Load video frames and extract corresponding audio segments efficiently.

    Args:
        video_path (str): Path to the video file.
        n_frms (int, optional): Number of frames to sample. Defaults to inf (all frames).
        height (int, optional): Height to resize the video frames to. Defaults to -1 (original height).
        width (int, optional): Width to resize the video frames to. Defaults to -1 (original width).
        sampling (str, optional): Sampling strategy for frame selection.
                                One of ["uniform", "random", "headtail"]. Defaults to "uniform".
        clip_proposal (tuple, optional): Start and end time (in seconds) to extract a clip from the video.
                                    Defaults to None (entire video).
        target_sr (int): Sampling rate to convert the audio to. Defaults to 48000.

    Returns:
        tuple: A tuple containing:
            - frms (torch.Tensor): Video frames of shape (C, T, H, W), where:
                - C: Number of color channels (e.g., 3 for RGB)
                - T: Number of sampled frames
                - H: Height of the frames
                - W: Width of the frames
            - indices (list[int]): List of selected frame indices.
            - fps (float): Frames per second of the video.
            - audio_segments (list[torch.Tensor]): List of audio tensors, where each tensor has shape (C, T ,T_audio):
                - C: Number of audio channels (e.g., 1 for mono, 2 for stereo)
                - T: Number of sampled audio clips
                - T_audio: Number of audio samples in the segment.
    """

    vr = VideoReader(uri=video_path, height=height, width=width)
    num_frames = len(vr)
    n_frms = min(n_frms, num_frames)
    fps = vr.get_avg_fps()

    audio = AudioReader(video_path, sample_rate=target_sr)
    sample_rate = audio.sample_rate
    print(f"audio: {audio.shape}")
    print(f"fps: {fps}")
    print(f"fps: {sample_rate}")

    if clip_proposal is None:
        start, end = 0, num_frames
    else:
        start, end = int(clip_proposal[0] * fps), int(clip_proposal[1] * fps)
        if start < 0:
            start = 0
        if end > num_frames:
            end = num_frames

    indices, intervals = sample_intervals_torch(sampling, start, end, n_frms, num_frames)

    frms = vr.get_batch(indices)
    frms = thwc_to_cthw(frms)

    # Calculate segment durations efficiently
    frame_lengths = intervals[:, 1] - intervals[:, 0]
    segment_duration = int(frame_lengths.sum() / len(intervals) * (1 / fps) * sample_rate)

        # Extract audio segments efficiently
    audio_start_indices = (intervals[:,0] / fps * sample_rate).int()
    audio_end_indices = torch.add(audio_start_indices,segment_duration)
    audio_seg_indicies = [torch.arange(start, end) for start, end in zip(audio_start_indices, audio_end_indices)]
    audio_segments = torch.stack([audio.get_batch(audio_indices) for audio_indices in audio_seg_indicies])
    audio_segments = audio_segments.permute(1,0,2).squeeze(0)

    return frms, indices.tolist(), fps, audio_segments[:n_frms], sample_rate

def thwc_to_cthw(data: torch.Tensor) -> torch.Tensor:
    """
    Permute tensor from (time, height, weight, channel) to
    (channel, height, width, time).
    """
    return data.permute(3, 0, 1, 2)


def resample_audio(waveform, orig_sr, target_sr):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    audio_waveform = resampler(waveform)  # Shape: [1, Resampled Samples]
    return audio_waveform

def sample_intervals_torch(sampling, start, end, n_frms, duration):
    """
    Sample a fixed number of intervals using torch.linspace and vectorized operations.

    Args:
        sampling (str): Sampling strategy ("uniform", "random", "headtail").
        start (int): Start index.
        end (int): End index.
        n_frms (int): Number of frames to sample.
        duration (int): Total duration of the sequence.

    Returns:
        tuple: A tuple containing:
            - indices (torch.Tensor): Tensor of selected frame indices.
            - ranges (torch.Tensor): Tensor of interval start and end indices.
    """

    # Create intervals using torch.linspace
    intervals = torch.linspace(start, end, steps=n_frms + 1, dtype=torch.long)

    # Calculate interval ranges
    ranges = torch.stack([intervals[:-1], intervals[1:]]).T

    # Sample frame indices based on sampling strategy
    if sampling == "uniform":
        indices = (ranges[:, 0] + ranges[:, 1]) // 2
    elif sampling == "random":
        indices = []
        for start, end in ranges:
            if start == end:
                indices.append(torch.tensor(start))
            else:
                indices.append(torch.randint(start, end, (1,)))
        indices = torch.cat(indices)
    elif sampling == "headtail":
        indices_h = torch.randperm(duration // 2)[:n_frms // 2]
        indices_t = torch.randperm(duration // 2, duration)[:n_frms // 2]
        indices = torch.cat((indices_h, indices_t))
    else:
        raise NotImplementedError

    # Ensure enough frames are sampled (not strictly necessary with vectorized operations)
    if len(indices) < n_frms:
        # Efficiently pad indices with the last element using tensor operations
        padding = torch.full((n_frms - len(indices),), indices[-1], dtype=torch.long)
        indices = torch.cat((indices, padding))

    return indices, ranges

def pad_or_random_crop(tensor, target_length, padding_value=0):
    """
    Pads or crops the features of a tensor to match the target length.

    Args:
        tensor (torch.Tensor): Input tensor of shape [20, features].
        target_length (int): The desired length of the features dimension.
        padding_value (float): The value to use for padding if features are shorter than the target_length.

    Returns:
        torch.Tensor: Tensor of shape [T, target_length].
    """
    num_rows, num_features = tensor.shape

    if num_features < target_length:
        # Pad the tensor along the features dimension
        pad_size = target_length - num_features
        padded_tensor = torch.nn.functional.pad(
            tensor, (0, pad_size), mode='constant', value=padding_value
        )
        return padded_tensor

    elif num_features > target_length:
        # Randomly crop a continuous sequence of target_length
        start_idx = torch.randint(0, num_features - target_length, (1,))
        cropped_tensor = tensor[:, start_idx : start_idx + target_length]
        return cropped_tensor

    else:
        # If already the target length, return the tensor as is
        return tensor
