import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def analyze_videos(video_folder):
    """Analyzes videos in a given folder, extracting duration and FPS.

    Args:
        video_folder (str): Path to the folder containing videos.

    Returns:
        list: List of tuples, each containing (duration, fps) for a video.
    """

    video_data = []
    for filename in tqdm(os.listdir(video_folder)):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Add more formats as needed
            filepath = os.path.join(video_folder, filename)
            cap = cv2.VideoCapture(filepath)

            # Get video properties efficiently
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            video_data.append((duration, fps, frame_count))
            cap.release()

    return video_data


def plot_distribution(data, title, xlabel, ylabel, output_path):
    """
    Plot the distribution of data.

    Args:
        data (list): Data values to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        output_path (str): Path to save the plot image.
    """
    plt.figure()
    plt.hist(data, bins=40, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    plt.close()

def main(folder_path, output_folder):
    """
    Main function to process videos and plot distributions.

    Args:
        folder_path (str): Path to the folder containing video files.
        output_folder (str): Path to the folder to save the plots.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_data = analyze_videos(folder_path)
    # Process videos in the folder
    durations, fps_values, frames = zip(*video_data)


    # Plot duration distribution
    if durations:
        plot_distribution(
            durations,
            title="Video Duration Distribution",
            xlabel="Duration (seconds)",
            ylabel="Frequency",
            output_path=os.path.join(output_folder, "cv2_duration_distribution.png")
        )

    # Plot FPS distribution
    if fps_values:
        plot_distribution(
            fps_values,
            title="Video FPS Distribution",
            xlabel="Frames Per Second (FPS)",
            ylabel="Frequency",
            output_path=os.path.join(output_folder, "cv2_fps_distribution.png")
        )
    # Plot FPS distribution
    if frames:
        plot_distribution(
            frames,
            title="Video Frames Distribution",
            xlabel="Frames",
            ylabel="Frequency",
            output_path=os.path.join(output_folder, "cv2_frames_distribution.png")
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze video metadata and plot distributions.")
    parser.add_argument("--videos", type=str, required=True, help="Path to the folder containing video files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save plots.")
    args = parser.parse_args()

    main(args.videos, args.output_folder)
    #main("C:\\Users\\tschesche\\Documents\\Uni\\PL-MAI\\Charades-STA\\Charades_v1","./")
