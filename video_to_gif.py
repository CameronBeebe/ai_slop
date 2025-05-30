import cv2
from PIL import Image
import os
import argparse

def video_to_gif(video_path, output_path, gif_fps=10):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return
    
    # Get the video's frame rate
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("Error: Could not determine video frame rate.")
        cap.release()
        return
    
    # Cap the GIF frame rate at the video's frame rate to avoid speeding up
    effective_gif_fps = min(gif_fps, video_fps)
    inclusion_rate = effective_gif_fps / video_fps
    
    # List to store PIL Images
    images = []
    s = 0  # Fractional counter for frame selection
    
    # Read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        s += inclusion_rate
        if s >= 1:
            # Convert frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image and add to list
            pil_image = Image.fromarray(rgb_frame)
            images.append(pil_image)
            s -= 1
    
    # Release the video capture object
    cap.release()
    
    # Check if any frames were extracted
    if not images:
        print("Error: No frames were extracted.")
        return
    
    # Calculate duration between frames in milliseconds
    duration = int(1000 / effective_gif_fps)
    
    # Save the GIF
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert a video to a GIF.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("output_path", help="Path to the output GIF file.")
    parser.add_argument("--fps", type=float, default=10, help="Frame rate of the GIF (default: 10).")
    args = parser.parse_args()
    
    # Call the function with provided arguments
    video_to_gif(args.video_path, args.output_path, args.fps)