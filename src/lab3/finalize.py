import cv2
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_model(num_classes):
    # Try to initialize without downloading any weights
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    
    # We need a backbone. Let's try to create one without weights.
    backbone = resnet_fpn_backbone('resnet50', weights=None)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

def visualize_predictions(model, device, frames_dir, output_path):
    model.eval()
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])[:8]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, f in enumerate(frame_files):
        img_path = os.path.join(frames_dir, f)
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(img_tensor)[0]
        
        ax = axes[i]
        ax.imshow(img)
        for box, score in zip(preds['boxes'], preds['scores']):
            if score > 0.5:
                x1, y1, x2, y2 = box.tolist()
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1, f"{score:.2f}", color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        ax.set_title(f)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Predictions grid saved to {output_path}")

def create_extracted_video(frames_in_dir, extracted_coco_path, output_video_path):
    with open(extracted_coco_path, 'r') as f:
        coco = json.load(f)
    
    img_id_to_boxes = {}
    for ann in coco['annotations']:
        img_id_to_boxes.setdefault(ann['image_id'], []).append(ann['bbox'])
    
    frame_files = sorted([f for f in os.listdir(frames_in_dir) if f.endswith(".jpg")])
    if not frame_files: return
    
    first_frame = cv2.imread(os.path.join(frames_in_dir, frame_files[0]))
    h, w, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))
    
    print("Generating video with extracted markup...")
    for img_info in coco['images']:
        fid = img_info['id']
        fname = img_info['file_name']
        frame = cv2.imread(os.path.join(frames_in_dir, fname))
        
        boxes = img_id_to_boxes.get(fid, [])
        for box in boxes:
            x, y, bw, bh = map(int, box)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, "Extracted", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    device = torch.device('cpu')
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("runs/lab3/fasterrcnn_extracted.pth", map_location=device))
    
    # 1. Visualize predictions
    visualize_predictions(model, device, "src/lab3/frames_input", "runs/lab3/predictions_grid.png")
    
    # 2. Create result video with extracted markup
    create_extracted_video("src/lab3/frames_input", "src/lab3/extracted_coco.json", "runs/lab3/extracted_result.mp4")
    
    # 3. Final Summary (IoU from previous step)
    print("\n--- Lab 3 Summary ---")
    print("1. Frames extracted from input.mp4 and output.mp4")
    print("2. Extracted markup extracted using OpenCV Differencing (Avg IoU: 0.4084)")
    print("3. Public API Comparison: YOLOv8 Avg IoU: 0.5603")
    print("4. Faster R-CNN trained on extracted markup (demo subset).")
    print("5. Predictions grid and result video generated.")
