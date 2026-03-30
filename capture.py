from PIL import Image
import cv2
from pypylon import pylon
import numpy as np
import os
import pickle
import torch
from torchvision import transforms


class DinoLivePredictor:
    def __init__(self, model_path, threshold=10.0):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = (252, 252)
        self.patch_size = 14

        print("Loading DINOv2 model...")
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.backbone.to(self.device)
        self.backbone.eval()

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.nbrs = data["nbrs"]

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _find_defect_bbox(self, anomaly_map_resized):
        norm_map = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        smooth_map = cv2.GaussianBlur(norm_map, (5, 5), 0)

        max_value = int(smooth_map.max())
        if max_value < 40:
            return None

        mean, std = cv2.meanStdDev(smooth_map)
        dynamic_thr = int(max(160, mean[0][0] + 1.25 * std[0][0], 0.85 * max_value))
        _, binary = cv2.threshold(smooth_map, dynamic_thr, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = smooth_map.shape
        min_area = max(16, int(0.001 * h * w))

        peak_y, peak_x = np.unravel_index(np.argmax(smooth_map), smooth_map.shape)
        best_bbox = None
        best_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if x <= peak_x <= x + bw and y <= peak_y <= y + bh:
                return (x, y, bw, bh)

            if area > best_area:
                best_area = area
                best_bbox = (x, y, bw, bh)

        return best_bbox

    def _detect_marker_stain_bbox(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

        b = frame_bgr[:, :, 0].astype(np.int16)
        g = frame_bgr[:, :, 1].astype(np.int16)
        r = frame_bgr[:, :, 2].astype(np.int16)

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        a_channel = lab[:, :, 1]

        red_excess = r - ((g + b) // 2)

        # Hybrid stain mask: catches low-saturation reddish stains as well as vivid marker blobs.
        mask_low_sat = (a_channel > 142) & (red_excess > 10) & (val > 35)
        mask_high_sat = (sat > 38) & (red_excess > 16) & (val > 35)
        mask = np.where(mask_low_sat | mask_high_sat, 255, 0).astype(np.uint8)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None

        h, w = frame_bgr.shape[:2]
        min_area = max(20, int(0.0002 * h * w))

        best_bbox = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            patch_red_excess = red_excess[y : y + bh, x : x + bw]
            patch_mask = mask[y : y + bh, x : x + bw] > 0
            if not np.any(patch_mask):
                continue

            mean_red = float(np.mean(patch_red_excess[patch_mask]))
            contour_score = area * max(mean_red, 0.0)

            if contour_score > best_score:
                best_score = contour_score
                best_bbox = (x, y, bw, bh)

        if best_bbox is None:
            return False, None

        return True, best_bbox

    def predict(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            ret = self.backbone.forward_features(input_tensor)
            patch_tokens = ret["x_norm_patchtokens"]

        h_feat = self.target_size[0] // self.patch_size
        w_feat = self.target_size[1] // self.patch_size
        features = patch_tokens[0].cpu().numpy()
        distances, _ = self.nbrs.kneighbors(features)
        anomaly_map = distances.reshape(h_feat, w_feat)

        frame_h, frame_w = frame_bgr.shape[:2]
        anomaly_map_resized = cv2.resize(anomaly_map, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        score = float(np.max(anomaly_map_resized))
        dino_defect = score > self.threshold
        dino_bbox = self._find_defect_bbox(anomaly_map_resized) if dino_defect else None

        stain_defect, stain_bbox = self._detect_marker_stain_bbox(frame_bgr)

        # Prefer stain localization when marker-like artifact is present.
        is_defect = dino_defect or stain_defect
        if stain_defect:
            defect_bbox = stain_bbox
        else:
            defect_bbox = dino_bbox

        return score, is_defect, defect_bbox, dino_defect, stain_defect


# -------------------- CAMERA INITIALIZATION --------------------
try:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    print(f"Camera initialized: {camera.GetDeviceInfo().GetModelName()}")

    try:
        if hasattr(camera, "GevSCPSPacketSize"):
            camera.GevSCPSPacketSize.SetValue(1200)
            print("Packet size set to 1200")

        if hasattr(camera, "GevSCPD"):
            camera.GevSCPD.SetValue(20000)
            print("Inter-Packet Delay set to 20000")

        if hasattr(camera, "AcquisitionFrameRateEnable"):
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(60)
            print("FPS limited to 60")

    except Exception as e:
        print(f"Could not apply camera settings: {e}")

except Exception as e:
    print(f"Camera initialization failed: {e}")
    raise SystemExit(1)

# -------------------- DINO LIVE PREDICTOR --------------------
MODEL_PATH = "dino_vits14.pkl"
THRESHOLD = 43.5

predictor = None
detection_enabled = False

try:
    predictor = DinoLivePredictor(MODEL_PATH, threshold=THRESHOLD)
    detection_enabled = True
    print(f"DINO live detection enabled. Threshold={THRESHOLD}")
except FileNotFoundError:
    print(f"DINO model artifact not found: {MODEL_PATH}")
    print("Live feed will run without defect detection.")
except Exception as e:
    print(f"Failed to initialize DINO predictor: {e}")
    print("Live feed will run without defect detection.")

# -------------------- START GRABBING --------------------
camera.MaxNumBuffer = 10
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# -------------------- CONVERTER --------------------
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# -------------------- SAVE DIRECTORY --------------------
save_dir = os.path.join(os.getcwd(), "captured_images")
os.makedirs(save_dir, exist_ok=True)

# -------------------- ROI --------------------
x_start, y_start = 663, 121
x_end, y_end = 1766, 1794

j = 100
last_score = 0.0
last_is_defect = False
last_bbox = None
last_dino_defect = False
last_stain_defect = False

# -------------------- MAIN LOOP --------------------
try:
    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)

        if grab_result is None:
            print("No frame received (timeout)")
            continue

        try:
            if not grab_result.GrabSucceeded():
                print(f"Grab failed: {grab_result.ErrorCode}")
                continue

            image = converter.Convert(grab_result)
            if image is None or not image.IsValid():
                print("Invalid image buffer, skipping frame")
                continue

            try:
                frame = image.GetArray()
            except Exception as e:
                print(f"GetArray error: {e}")
                continue

            if frame is None or frame.size == 0:
                print("Empty frame")
                continue

            h, w = frame.shape[:2]
            x1 = max(0, min(x_start, w))
            x2 = max(0, min(x_end, w))
            y1 = max(0, min(y_start, h))
            y2 = max(0, min(y_end, h))
            cropped_frame = frame[y1:y2, x1:x2]

            if cropped_frame.size == 0:
                print("Invalid ROI")
                continue

            display_frame = cropped_frame.copy()

            if detection_enabled and predictor is not None:
                try:
                    (
                        last_score,
                        last_is_defect,
                        last_bbox,
                        last_dino_defect,
                        last_stain_defect,
                    ) = predictor.predict(cropped_frame)
                except Exception as infer_error:
                    print(f"DINO inference error: {infer_error}")

            if detection_enabled and predictor is not None:
                status_text = "NOT OK" if last_is_defect else "OK"
                status_color = (0, 0, 255) if last_is_defect else (0, 200, 0)
                cv2.putText(
                    display_frame,
                    f"DINO: {status_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    status_color,
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Score: {last_score:.3f}  Th: {THRESHOLD:.3f}",
                    (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                reason_text = "Reason: marker stain" if last_stain_defect else ("Reason: DINO anomaly" if last_dino_defect else "Reason: none")
                cv2.putText(
                    display_frame,
                    reason_text,
                    (20, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255) if last_stain_defect else (255, 255, 255),
                    2,
                )
                if last_bbox is not None:
                    bx, by, bw, bh = last_bbox
                    cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                    cv2.putText(
                        display_frame,
                        "Defect region",
                        (bx, max(20, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
            else:
                cv2.putText(
                    display_frame,
                    "DINO: OFF (artifact missing or init failed)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 215, 255),
                    2,
                )

            cv2.putText(
                display_frame,
                "Keys: s=save, d=toggle detection, q=quit",
                (20, max(30, display_frame.shape[0] - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Cropped Live Feed", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                image_path = os.path.join(save_dir, f"cropped_r1{j}.jpg")
                success = cv2.imwrite(image_path, cropped_frame)
                if success:
                    print(f"Saved: {image_path}")
                else:
                    print("Failed to save image")
                j += 1

            if key == ord("d") and predictor is not None:
                detection_enabled = not detection_enabled
                print(f"DINO detection {'enabled' if detection_enabled else 'disabled'}")

            if key == ord("q"):
                print("Exit signal received")
                break

        except Exception as loop_error:
            print(f"Frame processing error: {loop_error}")
        finally:
            grab_result.Release()

except Exception as e:
    print(f"Unexpected error: {e}")

# -------------------- CLEANUP --------------------
finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    print("Camera stopped and resources released.")