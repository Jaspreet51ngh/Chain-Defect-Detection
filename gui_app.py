import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from torchvision import transforms
import pickle
import threading
import os
import sys

# Append src to path
sys.path.append(os.getcwd())
from src.segmentation import segment_jewelry

class DefectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chain Defect Detector (DINOv2)")
        self.root.geometry("1400x800")
        self.root.configure(bg="#2c3e50")

        # variables
        self.model = None
        self.nbrs = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.is_blinking = False
        self.blink_state = False
        self.threshold = 15.0 # DINO threshold based on report
        
        # UI Layout
        self._setup_ui()
        
        # Load Model in background
        self.status_label.config(text="Loading Model (DINOv2)... Please Wait")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _setup_ui(self):
        # Top Bar
        top_frame = tk.Frame(self.root, bg="#34495e", height=60)
        top_frame.pack(fill=tk.X)
        
        tk.Button(top_frame, text="Load Image", command=self.load_image, 
                  bg="#2980b9", fg="white", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20, pady=10)
        
        tk.Button(top_frame, text="Run Detection", command=self.run_detection, 
                  bg="#e67e22", fg="white", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20, pady=10)

        self.status_label = tk.Label(top_frame, text="Ready", bg="#34495e", fg="#ecf0f1", font=("Arial", 14))
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # Main Content (Split View)
        content_frame = tk.Frame(self.root, bg="#2c3e50")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left (Input)
        self.input_panel = tk.Label(content_frame, bg="black", text="Input Image", fg="gray")
        self.input_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Right (Output)
        self.output_panel = tk.Label(content_frame, bg="black", text="Output Result", fg="gray")
        self.output_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Indicator (Bottom)
        self.indicator_frame = tk.Frame(self.root, bg="#2c3e50", height=100)
        self.indicator_frame.pack(fill=tk.X, pady=10)
        
        self.indicator = tk.Label(self.indicator_frame, text="UNKNOWN", 
                                  bg="gray", fg="white", font=("Arial", 24, "bold"), width=20, height=2)
        self.indicator.pack()

    def _load_model(self):
        try:
            print("Loading DINOv2...")
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.backbone.to(self.device)
            self.backbone.eval()
            
            with open("dino_vits14.pkl", 'rb') as f:
                data = pickle.load(f)
                self.nbrs = data['nbrs']
                
            self.model_loaded = True
            self.root.after(0, lambda: self.status_label.config(text="Model Loaded (DINOv2) - Ready"))
            print("Model Loaded.")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}"))

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return
            
        self.current_image_path = file_path
        
        # Load and resize for display
        img = Image.open(file_path)
        
        # Aspect Ratio Resize logic
        # Max width 600, Max height 500
        w, h = img.size
        new_w = 600
        new_h = int((new_w / w) * h)
        
        self.display_img_pil = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_img_tk = ImageTk.PhotoImage(self.display_img_pil)
        
        self.input_panel.config(image=self.display_img_tk, text="")
        self.output_panel.config(image=None, text="Click 'Run Detection'")
        self._reset_indicator()

    def run_detection(self):
        if not self.model_loaded:
            messagebox.showwarning("Wait", "Model is still loading...")
            return
        if not hasattr(self, 'current_image_path'):
            messagebox.showwarning("Warning", "Load an image first.")
            return

        self.status_label.config(text="Processing...")
        self.root.update()
        
        try:
            # Inference Logic (same as predict_dino.py)
            image = Image.open(self.current_image_path).convert('RGB')
            w, h = image.size
            
            # --- Preprocessing ---
            # Segment mostly for removing background noise from metrics, 
            # but user wants robust box.
            # DINO + Masking is best.
            _, mask_orig = segment_jewelry(image)
            if mask_orig.sum() == 0: mask_orig = np.ones((h, w), dtype=np.float32)

            target_size = (252, 252)
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            # --- Forward Pass ---
            with torch.no_grad():
                ret = self.backbone.forward_features(input_tensor)
                patch_tokens = ret['x_norm_patchtokens']
            
            features = patch_tokens[0].cpu().numpy()
            distances, _ = self.nbrs.kneighbors(features)
            
            H_feat, W_feat = target_size[0] // 14, target_size[1] // 14
            anomaly_map = distances.reshape(H_feat, W_feat)
            
            # --- Post-Processing ---
            # Resize map to original image
            anomaly_map_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Apply Mask
            if mask_orig.shape != anomaly_map_resized.shape:
                mask_orig = cv2.resize(mask_orig, (w, h))
            
            masked_map = anomaly_map_resized * mask_orig
            score = np.max(masked_map)
            
            is_defect = score > self.threshold
            
            # --- Visualization ---
            # Prepare Output Image (Resized for Display)
            # Use same display size as input for consistency
            disp_w, disp_h = self.display_img_pil.size
            output_pil = self.display_img_pil.copy()
            draw_img = np.array(output_pil)
            
            if is_defect:
                # Dynamic Thresholding for Precision
                # We want to highlight only the "peak" anomalies (the actual defect), 
                # not the general noise of the chain.
                # Threshold = Max(BaseThreshold, 75% of MaxScore)
                dynamic_thresh = max(self.threshold, score * 0.75)
                
                # Threshold the map
                defect_mask = (masked_map > dynamic_thresh).astype(np.uint8) * 255
                
                # Close gaps (smaller kernel to avoid merging too much)
                kernel = np.ones((3,3), np.uint8)
                defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # We need to map coordinates from Original (w,h) to Display (disp_w, disp_h)
                scale_x = disp_w / w
                scale_y = disp_h / h
                
                box_count = 0
                for cnt in contours:
                    if cv2.contourArea(cnt) > 10: 
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        
                        # Scale rect
                        dx = int(x * scale_x)
                        dy = int(y * scale_y)
                        dw = int(cw * scale_x)
                        dh = int(ch * scale_y)
                        
                        # Add Padding (Reduced to 10 for tighter fit)
                        pad = 10
                        dx = max(0, dx - pad)
                        dy = max(0, dy - pad)
                        dw = min(disp_w - dx, dw + 2*pad)
                        dh = min(disp_h - dy, dh + 2*pad)
                        
                        # Draw Box (Red, Thick)
                        cv2.rectangle(draw_img, (dx, dy), (dx+dw, dy+dh), (255, 0, 0), 3)
                        cv2.putText(draw_img, "DEFECT", (dx, dy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        box_count += 1
                
                # Fallback: If score says defect but no contours found
                if box_count == 0:
                     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(masked_map)
                     mx, my = maxLoc
                     
                     dmx = int(mx * scale_x)
                     dmy = int(my * scale_y)
                     
                     # Draw Crosshair
                     cv2.circle(draw_img, (dmx, dmy), 20, (255, 0, 0), 3)
                     cv2.line(draw_img, (dmx-10, dmy), (dmx+10, dmy), (255, 0, 0), 2)
                     cv2.line(draw_img, (dmx, dmy-10), (dmx, dmy+10), (255, 0, 0), 2)
                     cv2.putText(draw_img, f"MAX: {score:.1f}", (dmx+25, dmy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Fallback: If score says defect but no contours found (e.g. single pixel spike),
                # Draw circle at max point
                if box_count == 0:
                     minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(masked_map)
                     mx, my = maxLoc
                     
                     dmx = int(mx * scale_x)
                     dmy = int(my * scale_y)
                     
                     # Draw Crosshair
                     cv2.circle(draw_img, (dmx, dmy), 20, (255, 0, 0), 3)
                     cv2.line(draw_img, (dmx-10, dmy), (dmx+10, dmy), (255, 0, 0), 2)
                     cv2.line(draw_img, (dmx, dmy-10), (dmx, dmy+10), (255, 0, 0), 2)
                     cv2.putText(draw_img, f"MAX: {score:.1f}", (dmx+25, dmy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Update Output Panel
            self.output_tk = ImageTk.PhotoImage(Image.fromarray(draw_img))
            self.output_panel.config(image=self.output_tk, text="")
            
            # Update Indicator
            self._set_indicator(is_defect)
            self.status_label.config(text=f"Done using DINOv2")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()

    def _reset_indicator(self):
        self.is_blinking = False
        self.indicator.config(text="UNKNOWN", bg="gray")

    def _set_indicator(self, is_defect):
        self.is_blinking = True
        self.blink_color = "red" if is_defect else "green"
        self.blink_text = "DEFECT FOUND" if is_defect else "OK"
        self._blink_loop()

    def _blink_loop(self):
        if not self.is_blinking: return
        
        current_bg = self.indicator.cget("bg")
        new_bg = self.blink_color if current_bg == "black" else "black"
        
        self.indicator.config(text=self.blink_text, bg=new_bg)
        self.root.after(500, self._blink_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = DefectDetectionApp(root)
    root.mainloop()
