import os
import pickle

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            files.append(os.path.join(folder, name))
    return files


def score_image(path, backbone, nbrs, transform, device):
    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        ret = backbone.forward_features(input_tensor)
        patch_tokens = ret["x_norm_patchtokens"]
    features = patch_tokens[0].cpu().numpy()
    distances, _ = nbrs.kneighbors(features)
    anomaly_map = distances.reshape(18, 18)
    return float(np.max(anomaly_map))


def collect_scores(paths, backbone, nbrs, transform, device):
    scores = []
    for path in paths:
        try:
            scores.append(score_image(path, backbone, nbrs, transform, device))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return np.array(scores, dtype=np.float32)


def summarize(name, arr):
    if arr.size == 0:
        print(f"{name}: no images")
        return
    print(f"{name}: count={arr.size}")
    print(
        f"{name} min/mean/max = {arr.min():.4f} / {arr.mean():.4f} / {arr.max():.4f}"
    )
    print(f"{name} p95/p99 = {np.percentile(arr,95):.4f} / {np.percentile(arr,99):.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "dino_vits14.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        nbrs = data["nbrs"]

    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.to(device)
    backbone.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((252, 252)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    good_paths = list_images("dataset/train/good")
    bad_paths = list_images("dataset/test/bad")
    synthetic_paths = list_images("dataset/test/synthetic")

    good_scores = collect_scores(good_paths, backbone, nbrs, transform, device)
    bad_scores = collect_scores(bad_paths, backbone, nbrs, transform, device)
    synthetic_scores = collect_scores(synthetic_paths, backbone, nbrs, transform, device)

    summarize("GOOD", good_scores)
    summarize("BAD", bad_scores)
    summarize("SYNTHETIC", synthetic_scores)

    if good_scores.size > 0:
        suggested = float(np.percentile(good_scores, 99.5))
        print(f"Suggested threshold (good p99.5): {suggested:.4f}")


if __name__ == "__main__":
    main()
