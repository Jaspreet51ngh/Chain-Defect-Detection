import argparse
import torch
import os
from torch.utils.data import DataLoader
from src.dataset import ChainDataset
from src.model import PatchCore
from src.utils import plot_anomaly
from PIL import Image
from torchvision import transforms

def train(args):
    dataset = ChainDataset(root_dir=args.data_path)
    if len(dataset) == 0:
        print(f"No images found in {args.data_path}. Please add images to dataset/train/good/")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = PatchCore(backbone_name="resnet50", memory_bank_size=args.bank_size)
    model.fit(dataloader)
    
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

def predict(args):
    model = PatchCore(backbone_name="resnet50")
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}. Please train first.")
        return
    model.load(args.model_path)
    
    # Load and preprocess single image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Batch dim
    
    score, amap = model.predict(input_tensor)
    
    result = "NOT OK" if score > args.threshold else "OK"
    print(f"{{'anomaly_score': {score:.4f}, 'prediction': '{result}'}}")
    
    if args.vis:
        plot_anomaly(input_tensor[0], amap, score, args.threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Chain Defect Detection PoC")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # Train Parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_path", type=str, default="dataset/train/good")
    train_parser.add_argument("--model_path", type=str, default="model.pkl")
    train_parser.add_argument("--bank_size", type=int, default=1000)
    
    # Predict Parser
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--image_path", type=str, required=True)
    predict_parser.add_argument("--model_path", type=str, default="model.pkl")
    predict_parser.add_argument("--threshold", type=float, default=7.85) # Tuned based on calibration (Max Good 7.12)
    predict_parser.add_argument("--vis", action="store_true", help="Visualize output")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
