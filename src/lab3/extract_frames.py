import cv2
import os

def extract_frames(video_path, output_dir, max_frames=301):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames from {video_path} to {output_dir}")

if __name__ == "__main__":
    extract_frames("input.mp4", "src/lab3/frames_input")
    extract_frames("output.mp4", "src/lab3/frames_output")
