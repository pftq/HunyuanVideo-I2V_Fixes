"""
20250324 pftq:
This is a script for easily converting your data set for use with diffusion pipeline (for T2V, Wan, etc).
- Reads from input_dir and writes to output_dir to avoid I/O issues
- Renames all files (json and mp4) to subfolder_original-filename
- Moves all files out of subfolders to the root of output_dir
- Renames JSON files to .txt and retains only the 'long caption' values

Configure settings below.
"""

import os
import json
import shutil
from pathlib import Path

# Configuration
input_dir = "training"           # Directory containing the input dataset (videos and JSON files)
output_dir = "training_diffusionpipe"     # Directory where the processed dataset will be written

def flatten_and_convert_directory(input_dir, output_dir):
    """Flatten directory, rename files, and convert JSON to TXT with long captions."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define file extensions
    media_extensions = (".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png")
    json_extension = ".json"
    
    print("Processing files in directory...")
    files_to_process = []
    
    # Step 1: Collect all files and prepare new names
    for file in input_path.rglob("*"):  # Recursively search all subfolders in input_dir
        if file.is_file():  # Ensure it's a file, not a directory
            relative_path = file.relative_to(input_path)  # e.g., subfolder1/video_1.mp4
            parts = relative_path.parts  # Split into parts: ('subfolder1', 'video_1.mp4')
            
            # Construct new filename: subfolder_original-filename
            if len(parts) > 1:
                # File is in a subfolder
                subfolder = "_".join(parts[:-1])  # Join all parts except the filename (e.g., subfolder1_subfolder2)
                original_filename = parts[-1]  # e.g., video_1.mp4
                new_filename = f"{subfolder}_{original_filename}"  # e.g., subfolder1_video_1.mp4
            else:
                # File is already in the root
                new_filename = parts[0]  # e.g., video_1.mp4
            
            # New path in the output directory (root)
            new_path = output_path / new_filename
            files_to_process.append((file, relative_path, new_path))
    
    # Step 2: Copy and rename files to output_dir
    for file, relative_path, new_path in files_to_process:
        if file.suffix.lower() in media_extensions or file.suffix.lower() == json_extension:
            print(f"Copying and renaming {relative_path} to {new_path.name}")
            shutil.copy2(file, new_path)  # Use copy2 to preserve metadata
    
    # Step 3: Process files in output_dir (convert JSON to TXT)
    files_to_process = []
    for file in output_path.glob("*"):  # Only look in the root of output_dir
        if file.is_file():
            relative_path = file.relative_to(output_path)
            files_to_process.append((file, relative_path))
    
    for file, relative_path in files_to_process:
        if file.suffix.lower() == json_extension:
            # Process JSON file: extract long caption and convert to .txt
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # Extract the "long caption" field
                long_caption = data.get("raw_caption", {}).get("long caption", "")
                if not long_caption:
                    print(f"Warning: No 'long_caption' found in {relative_path}, using empty string")
                
                # New filename: change extension to .txt
                new_filename = file.stem + ".txt"  # e.g., video_1.txt
                new_path = output_path / new_filename
                
                # Write the long caption to the new .txt file
                print(f"Converting {relative_path} to {new_filename} with long caption")
                with open(new_path, 'w') as f:
                    f.write(long_caption)
                
                # Remove the original JSON file
                file.unlink()
            
            except json.JSONDecodeError as e:
                print(f"Error: Could not parse JSON in {relative_path}: {e}")
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")

    print("Dataset processing complete!")

def main():
    flatten_and_convert_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()
