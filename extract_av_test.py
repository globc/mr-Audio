import os
import csv
import argparse
from audioinclusion.intervals import load_video_frames_with_audio

def process_videos(input_dir, output_csv):
    """
    Process all videos in the input directory and write their audio/video tensor shapes to a CSV file.

    Args:
        input_dir (str): Directory containing video files.
        output_csv (str): Path to output CSV file.
    """
    # Supported video formats
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
    
    # Prepare the CSV output file
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Video Name", "Video Tensor Shape", "Audio Tensor Shape"])

        # Loop through files in the input directory
        for video_file in os.listdir(input_dir):
            if os.path.splitext(video_file)[1].lower() in video_extensions:
                video_path = os.path.join(input_dir, video_file)
                try:
                    print(f"Processing: {video_file}")
                    # Read video and audio tensors
                    clip, indices, fps, audio, sr = load_video_frames_with_audio(
                        video_path=video_path,
                        n_frms=20,
                        height=224,
                        width=224,
                        sampling="random",
                        clip_proposal=None
                    )
                    
                    # Extract shapes
                    video_shape = clip.shape  # [Frames, Height, Width, Channels]
                    audio_shape = audio.shape  # [Audio Samples, Channels]

                    # Write to CSV
                    writer.writerow([video_file, str(video_shape), str(audio_shape)])
                except Exception as e:
                    print(f"Failed to process {video_file}: {e}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract video and audio tensor shapes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing video files.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    # Run the video processing function
    process_videos(args.input_dir, args.output_csv)

