import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import json
import time

def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_by_frame = {}
    for track in root.findall("track"):
        label = track.get("label")
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            fid = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            # COCO format: [x, y, w, h]
            gt_by_frame.setdefault(fid, []).append({
                "bbox": [xtl, ytl, xbr - xtl, ybr - ytl],
                "label": label
            })
    return gt_by_frame

def extract_opencv_boxes(in_img, out_img):
    diff = cv2.absdiff(in_img, out_img)
    mask = np.any(diff > 50, axis=-1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20: # Filter small noise
            boxes.append({"bbox": [float(x), float(y), float(w), float(h)], "label": "detected"})
    return boxes

def get_boxes_yolo(model, image, conf=0.3):
    results = model(image, verbose=False)[0]
    boxes = []
    # COCO classes for vehicles: 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASSES = [2, 3, 5, 7]
    for box in results.boxes:
        if int(box.cls) in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "label": model.names[int(box.cls)],
                "score": float(box.conf)
            })
    return boxes

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_method(preds, gts):
    if not gts: return 0.0
    total_iou = 0
    matched = 0
    for gt in gts:
        best_iou = 0
        for pred in preds:
            iou = calculate_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou = iou
        if best_iou > 0.3: # Match threshold
            total_iou += best_iou
            matched += 1
    return total_iou / len(gts)

if __name__ == "__main__":
    gt_data = parse_cvat_xml("annotations.xml")
    yolo_model = YOLO("yolov8n.pt") # Using nano for speed in lab environment
    
    frames_in = sorted([os.path.join("src/lab3/frames_input", f) for f in os.listdir("src/lab3/frames_input") if f.endswith(".jpg")])
    frames_out = sorted([os.path.join("src/lab3/frames_output", f) for f in os.listdir("src/lab3/frames_output") if f.endswith(".jpg")])
    
    results = []
    extracted_annotations = []
    
    print("Processing frames...")
    for i in range(min(len(frames_in), 301)):
        img_in = cv2.imread(frames_in[i])
        img_out = cv2.imread(frames_out[i])
        fid = int(os.path.basename(frames_in[i]).split("_")[1].split(".")[0])
        
        # Method 1: OpenCV Stealing
        t0 = time.time()
        cv_boxes = extract_opencv_boxes(img_in, img_out)
        t_cv = time.time() - t0
        
        # Method 2: YOLO API
        t0 = time.time()
        yolo_boxes = get_boxes_yolo(yolo_model, img_in)
        t_yolo = time.time() - t0
        
        gt = gt_data.get(fid, [])
        iou_cv = evaluate_method(cv_boxes, gt)
        iou_yolo = evaluate_method(yolo_boxes, gt)
        
        results.append({
            "frame": fid,
            "iou_cv": iou_cv,
            "iou_yolo": iou_yolo,
            "time_cv": t_cv,
            "time_yolo": t_yolo
        })
        
        # Store extracted markup (Method 1) for COCO
        extracted_annotations.append({
            "frame_id": fid,
            "file_name": os.path.basename(frames_in[i]),
            "boxes": cv_boxes,
            "width": img_in.shape[1],
            "height": img_in.shape[0]
        })
        
        if i % 50 == 0:
            print(f"Frame {i}: IoU CV={iou_cv:.3f}, IoU YOLO={iou_yolo:.3f}")

    # Summarize
    avg_iou_cv = np.mean([r["iou_cv"] for r in results])
    avg_iou_yolo = np.mean([r["iou_yolo"] for r in results])
    print(f"\nFinal Comparison:")
    print(f"OpenCV Difference (Stealing): Avg IoU = {avg_iou_cv:.4f}")
    print(f"YOLOv8 (Public API):         Avg IoU = {avg_iou_yolo:.4f}")

    # Save extracted markup to COCO
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "car"}]
    }
    
    ann_id = 1
    for item in extracted_annotations:
        coco_dataset["images"].append({
            "id": item["frame_id"],
            "file_name": item["file_name"],
            "width": item["width"],
            "height": item["height"]
        })
        for box in item["boxes"]:
            coco_dataset["annotations"].append({
                "id": ann_id,
                "image_id": item["frame_id"],
                "category_id": 1,
                "bbox": box["bbox"],
                "area": box["bbox"][2] * box["bbox"][3],
                "iscrowd": 0
            })
            ann_id += 1
            
    with open("src/lab3/extracted_coco.json", "w") as f:
        json.dump(coco_dataset, f, indent=2)
    print("\nSaved extracted markup to src/lab3/extracted_coco.json")
