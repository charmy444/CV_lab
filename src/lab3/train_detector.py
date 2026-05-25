import os
import json
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ExtractedDataset(torch.utils.data.Dataset):
    def __init__(self, coco_path, frames_dir):
        with open(coco_path, 'r') as f:
            self.coco = json.load(f)
        self.frames_dir = frames_dir
        self.image_ids = [img['id'] for img in self.coco['images']]
        self.id_to_img = {img['id']: img for img in self.coco['images']}
        self.id_to_anns = {}
        for ann in self.coco['annotations']:
            self.id_to_anns.setdefault(ann['image_id'], []).append(ann)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_info = self.id_to_img[img_id]
        path = os.path.join(self.frames_dir, img_info['file_name'])
        img = Image.open(path).convert("RGB")
        
        anns = self.id_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(1) # Only one category: car
        
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        img_tensor = F.to_tensor(img)
        return img_tensor, target

    def __len__(self):
        return len(self.image_ids)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    dataset = ExtractedDataset("src/lab3/extracted_coco.json", "src/lab3/frames_input")
    # Extremely small subset for the sake of the environment
    indices = torch.randperm(len(dataset)).tolist()[:5]
    dataset = torch.utils.data.Subset(dataset, indices)
    
    # Split
    train_ds, val_ds = dataset, dataset # Use same for demo
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    model = get_model(num_classes=2)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
    
    num_epochs = 1
    print(f"Starting extremely short training on {len(train_ds)} images...")
    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            print(f"  Processing batch {i+1}/{len(train_loader)}...", flush=True)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(f"    Batch {i+1} loss: {losses.item():.4f}", flush=True)

    torch.save(model.state_dict(), "runs/lab3/fasterrcnn_extracted.pth")
    print("Model saved to runs/lab3/fasterrcnn_extracted.pth")
    
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig("runs/lab3/loss_curve.png")
    print("Loss curve saved to runs/lab3/loss_curve.png")
