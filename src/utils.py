import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

def plot_anomaly(image, anomaly_map, score, threshold=None, save_path=None):
    """
    Overlays anomaly map on the original image.
    """
    img = denormalize(image)
    img = (img * 255).astype(np.uint8)
    
    # Resize anomaly map to image size
    anomaly_map = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]))
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(anomaly_map * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    status = "NOT OK" if threshold and score > threshold else "OK" if threshold else "Score: {:.4f}".format(score)
    plt.title(f"Anomaly Map | {status}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
