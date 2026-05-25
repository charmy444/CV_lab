import cv2
import numpy as np
import os

def analyze_diff(idx):
    in_img = cv2.imread(f"src/lab3/frames_input/frame_{idx:05d}.jpg")
    out_img = cv2.imread(f"src/lab3/frames_output/frame_{idx:05d}.jpg")
    if in_img is None or out_img is None:
        print("Frames not found")
        return
    
                                            
                                                
    
                            
    diff = cv2.absdiff(in_img, out_img)
    mask = np.any(diff > 50, axis=-1).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Frame {idx}: found {len(contours)} potential boxes via differencing")
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            print(f"  Box {i}: x={x}, y={y}, w={w}, h={h}")

if __name__ == "__main__":
    analyze_diff(50)                        
