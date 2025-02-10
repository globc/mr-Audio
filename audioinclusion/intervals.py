import torch
import torchvision.io
import torchaudio
import math

def load_video_frames_with_audio(video_path,
                                      n_frms=float('inf'),
                                      height=-1,
                                      width=-1,
                                      sampling="uniform",
                                      clip_proposal=None, target_sr=48000, audio_clip_len=1.0):
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
        target_sr (int): Target sample rate of the audio
        audio_clip_len (float): Length of the extracted audio clips, in seconds. Defaults to 1 second.

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

    indices, intervals = sample_intervals(sampling, start, end, n_frms, num_frames)

    # Extract video frames
    frms = torch.index_select(video, 0, indices)
    frms = thwc_to_cthw(frms)
    
    audio_segments=sample_audio_fixed_center(audio, intervals, fps, sample_rate, audio_clip_len)

    if height > 0 and width > 0:
        frms = torch.nn.functional.interpolate(frms, size=(height, width), mode='bilinear', align_corners=False)

    return frms, indices.tolist(), fps, torch.stack(audio_segments, dim=1).squeeze(0)

def thwc_to_cthw(data: torch.Tensor) -> torch.Tensor:
    """
    Permute tensor from (time, height, weight, channel) to
    (channel, height, width, time).
    """
    return data.permute(3, 0, 1, 2)

def sample_audio_fixed_center(audio, frame_indices, fps, sr, desired_length):
    # Convert frame time stamps to audio fram time stamps => [0, 30] -> [0, 48000] if sr = 48000
    start_times= ((frame_indices[:,0]/fps) * sr).int()
    end_times= ((frame_indices[:,1]/fps) * sr).int()

    duration = end_times - start_times
    audio_len = audio.shape[1]
    
    start_indices = torch.clamp(start_times, min=0, max=audio_len)
    end_indices = torch.clamp(end_times, min=0, max=audio_len)


    # Calculate the desired number of samples
    num_samples = int(desired_length * sr)

    # middle_frame + samples_left, middle_frame + samples_right
    dl = math.floor(num_samples / 2)
    dr = math.ceil(num_samples / 2)

    # split the audio into segments
    audio_segments = []
    for start, end in zip(start_indices, end_indices):
        duration = end - start
        #print(f"Segment: {start}, {end}")
        if(duration == 0):
            # If duration is zero, the start and end of the interval were outside the audio tensor
            segment = torch.zeros(audio.shape[0], num_samples)
        else:
            # The interval is [start,end], so the middle frame is ( (end-start)//2) + start
            middle = ((end - start) // 2) + start
            
            start_idx = middle - dl
            end_idx = middle + dr
            #print(f"clip interval: {start_idx},{end_idx}")
            segment = audio[:,start_idx:end_idx]
            segment = pad_or_crop_clip(segment, num_samples)

        audio_segments.append(segment)
    
    return  audio_segments
    

def resample_audio(waveform, orig_sr, target_sr):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    audio_waveform = resampler(waveform)  # Shape: [1, Resampled Samples]
    return audio_waveform

def sample_intervals(sampling, start, end, n_frms, duration):
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

def pad_or_crop_clip(tensor, target_length, padding_value=0):
    """
    Pads or crops the features of a tensor to match the target length.

    Args:
        tensor (torch.Tensor): Input tensor of shape [20, features].
        target_length (int): The desired length of the features dimension.
        padding_value (float): The value to use for padding if features are shorter than the target_length.

    Returns:
        torch.Tensor: Tensor of shape [T, target_length].
    """
    _, num_features = tensor.shape

    if num_features < target_length:
        # Pad the tensor along the features dimension
        pad_size = target_length - num_features
        padded_tensor = torch.nn.functional.pad(
            tensor, (0, pad_size), mode='constant', value=padding_value
        )
        return padded_tensor

    elif num_features > target_length:
        cropped_tensor = tensor[:, :target_length]
        return cropped_tensor

    else:
        # If already the target length, return the tensor as is
        return tensor
