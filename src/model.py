import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

class PatchCore:
    def __init__(self, backbone_name="resnet50", memory_bank_size=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_bank_size = memory_bank_size
        
        # Load backbone
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            self.layers_to_extract = ['layer2', 'layer3']
        else:
            raise NotImplementedError("Only ResNet50 supported for this PoC")
        
        self.backbone.to(self.device)
        self.backbone.eval()
        
        self.features = []
        self._register_hooks()
        
        self.memory_bank = None
        self.nbrs = None

    def _register_hooks(self):
        def hook(module, input, output):
            self.features.append(output)
        
        for name, module in self.backbone.named_children():
            if name in self.layers_to_extract:
                module.register_forward_hook(hook)

    def _embed(self, images):
        self.features = []
        with torch.no_grad():
            _ = self.backbone(images.to(self.device))
        
        # self.features is a list of tensors from layer2, layer3
        # Resizing to match the largest spatial resolution
        ref_h, ref_w = self.features[0].shape[2], self.features[0].shape[3]
        
        embeddings = []
        for feat in self.features:
            # Interpolate to match the first feature map size
            feat = F.interpolate(feat, size=(ref_h, ref_w), mode='bilinear', align_corners=True)
            embeddings.append(feat)
            
        # Concatenate along channel dimension
        embedding = torch.cat(embeddings, dim=1) 
        return embedding

    def fit(self, dataloader):
        print("Extracting features from training data...")
        embedding_list = []
        
        for images, _ in tqdm(dataloader):
            # shape: (B, C, H, W) -> (B, D, H', W')
            emb = self._embed(images)
            # shape: (B, H', W', D)
            emb = emb.permute(0, 2, 3, 1).contiguous()
            emb = emb.reshape(-1, emb.shape[-1]) # Flatten to (N_patches, D)
            embedding_list.append(emb.cpu())
            
        full_embeddings = torch.cat(embedding_list, dim=0)
        
        # Random Subsampling (Simple CoreSet approximation for PoC)
        print(f"Total patches: {full_embeddings.shape[0]}. Subsampling to {self.memory_bank_size}...")
        if full_embeddings.shape[0] > self.memory_bank_size:
            indices = np.random.choice(full_embeddings.shape[0], self.memory_bank_size, replace=False)
            self.memory_bank = full_embeddings[indices]
        else:
            self.memory_bank = full_embeddings
            
        print("Building Nearest Neighbors index...")
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=2)
        self.nbrs.fit(self.memory_bank.numpy())
        print("Training complete.")

    def predict(self, image_tensor):
        # image_tensor shape: (1, C, H, W)
        emb = self._embed(image_tensor)
        # emb shape: (1, D, H', W')
        B, D, H_prime, W_prime = emb.shape
        
        emb_flat = emb.permute(0, 2, 3, 1).reshape(-1, D).cpu().numpy()
        
        # Find distance to nearest neighbor in memory bank
        distances, _ = self.nbrs.kneighbors(emb_flat)
        
        # Reshape distances back to spatial map
        anomaly_map = distances.reshape(H_prime, W_prime)
        
        # Anomaly score is the maximum distance in the map (simplest aggregation)
        anomaly_score = np.max(anomaly_map)
        
        return anomaly_score, anomaly_map

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'memory_bank': self.memory_bank, 'nbrs': self.nbrs}, f)
            
    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.memory_bank = data['memory_bank']
            self.nbrs = data['nbrs']
