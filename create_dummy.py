from PIL import Image
import numpy as np
import os

os.makedirs('dataset/train/good', exist_ok=True)

# Create a simple gradient image
w, h = 256, 256
img = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img[i, j] = [i % 255, j % 255, (i+j) % 255]

image = Image.fromarray(img)
image.save('dataset/train/good/dummy.png')
print("Created dataset/train/good/dummy.png")
