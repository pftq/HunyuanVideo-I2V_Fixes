"""
20250322 pftq: 
Useful script for stretching or padding your videos to 129 frames to meet the training script requirements.
Configure settings below.

Make sure you have ffmpeg and ffmpeg-python installed.
On Linux, it is:
apt-get update
apt-get install -y ffmpeg
pip install ffmpeg-python
"""

import os
import subprocess
import ffmpeg
from pathlib import Path

# Configuration
input_dir = "training_original"  # Directory with your original videos
output_dir = "training"          # Overwrite in same directory (or change to a new one)
target_frames = 129              # Desired frame count
target_fps = 24                  # Output frame rate
target_duration = target_frames / target_fps  # 5.375 seconds
target_bitrate = "7M"            # 5 Mbps (5000 kbps)
stretchFrameRate = True          # True: stretch frame rate; False: pad with repeated frames

def get_video_info(file_path):
    """Get frame count and frame rate of a video using ffprobe."""
    try:
        probe = ffmpeg.probe(file_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        if not video_stream:
            raise ValueError("No video stream found")
        
        if "nb_frames" in video_stream:
            frame_count = int(video_stream["nb_frames"])
        else:
            duration = float(probe["format"]["duration"])
            fps = eval(video_stream["r_frame_rate"])
            frame_count = int(duration * fps)
        
        fps = eval(video_stream["r_frame_rate"])
        return frame_count, fps
    except ffmpeg.Error as e:
        print(f"Error probing {file_path}: {e.stderr.decode()}")
        return None, None

def adjust_video_to_frames(input_path, output_path, current_frames, target_frames, target_fps, target_duration, target_bitrate, stretchFrameRate):
    """Adjust video to exactly target_frames at target_fps with specified bitrate, either by stretching or padding."""
    try:
        # Common FFmpeg stream setup
        stream = ffmpeg.input(input_path)

        if stretchFrameRate:
            # Stretch mode: Adjust frame rate to stretch video to target_frames
            original_duration = current_frames / target_fps
            speed_factor = original_duration / target_duration
            pts_factor = 1 / speed_factor

            stream = stream.filter("setpts", f"{pts_factor}*PTS")  # Adjust speed to stretch
            stream = stream.filter("fps", fps=target_fps)          # Set target frame rate
            print(f"Stretching {input_path} from {current_frames} to {target_frames} frames "
                  f"(speed factor: {speed_factor:.2f}x, bitrate: {target_bitrate}, duration: {target_duration}s)")
        else:
            # Pad mode: Keep original frame rate and timing, pad with repeated frames to reach target_frames
            pad_frames = target_frames - current_frames  # Number of frames to pad
            if pad_frames < 0:
                # If video is longer than target_frames, trim it
                stream = stream.filter("trim", end_frame=target_frames)
                pad_frames = 0
            stream = stream.filter("fps", fps=target_fps)  # Ensure target frame rate (but no stretching)
            if pad_frames > 0:
                stream = stream.filter("tpad", stop_mode="clone", stop_duration=pad_frames/target_fps)  # Pad by repeating last frame
            print(f"Padding {input_path} from {current_frames} to {target_frames} frames "
                  f"(padding {pad_frames} frames, bitrate: {target_bitrate}, duration: {target_duration}s)")

        # Output settings (common to both modes)
        stream = stream.output(
            output_path,
            r=target_fps,
            vcodec="libx264",
            **{"b:v": target_bitrate},
            acodec="aac",
            map_metadata=0,
            t=target_duration,
            **{"frames:v": target_frames},
            y=None
        )
        stream.run(quiet=True)  # Verbose output for debugging
        
    except ffmpeg.Error as e:
        print(f"Error processing {input_path}: {e.stderr.decode()}")

def main():
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    for file in input_path.glob("*"):
        if file.suffix.lower() in video_extensions:
            input_file = str(file)
            output_file = str(output_path / file.name)
            
            frame_count, fps = get_video_info(input_file)
            if frame_count is None:
                continue
            
            if frame_count != target_frames:
                print(f"Processing {file.name}: {frame_count} frames -> {target_frames} frames")
                adjust_video_to_frames(input_file, output_file, frame_count, target_frames, target_fps, target_duration, target_bitrate, stretchFrameRate)
            else:
                print(f"{file.name} already has {target_frames} frames, skipping")

    print("All videos processed!")

if __name__ == "__main__":
    try:
        import ffmpeg
    except ImportError:
        print("Installing ffmpeg-python...")
        subprocess.run(["pip", "install", "ffmpeg-python"], check=True)
    
    main()
