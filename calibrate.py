import os
import glob
import argparse
from src.model import PatchCore
from src.dataset import ChainDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def calibrate(args):
    model = PatchCore(backbone_name="resnet50")
    if not os.path.exists(args.model_path):
        print("Model not found!")
        return
    model.load(args.model_path)
    
    image_paths = sorted(glob.glob(os.path.join(args.data_path, '**', '*.[jJ][pP][gG]'), recursive=True) +
                         glob.glob(os.path.join(args.data_path, '**', '*.[pP][nN][gG]'), recursive=True))
    
    print(f"Found {len(image_paths)} images in {args.data_path}")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    scores = []
    print(f"{'Image':<30} | {'Score':<10}")
    print("-" * 45)
    
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        score, _ = model.predict(input_tensor)
        scores.append(score)
        print(f"{os.path.basename(img_path):<30} | {score:.4f}")
        
    print("-" * 45)
    max_score = max(scores)
    print(f"Max Good Score: {max_score:.4f}")
    print(f"Recommended Threshold: {max_score * 1.1:.4f} (Max + 10%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/train/good")
    parser.add_argument("--model_path", type=str, default="model.pkl")
    args = parser.parse_args()
    calibrate(args)
