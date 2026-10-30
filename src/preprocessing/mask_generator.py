import os
import cv2
import torch
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

DYNAMIC_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}


class DynamicObjectMasker:
    def __init__(self, model_name="yolov8n-seg.pt", confidence=0.35, device=None):
        if YOLO is None:
            raise ImportError("[-] 'ultralytics' not installed. Run: pip install ultralytics")
        self.confidence = confidence
        
        # Bulletproof fallback for RTX 5060 (sm_120)
        self.device = "cpu"
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            try:
                cap_major, cap_minor = torch.cuda.get_device_capability(0)
                if cap_major > 9:
                    print(f"[!] Unsupported CUDA architecture sm_{cap_major}{cap_minor} detected for this PyTorch version. Forcing YOLO to CPU.")
                    self.device = "cpu"
                else:
                    self.device = 0
            except Exception:
                self.device = "cpu"

        print(f"[+] Loading YOLOv8 segmentation model ({model_name}) on device={self.device}...")
        self.model = YOLO(model_name)

    def generate_masks(
        self,
        images_dir: str,
        masks_dir: str,
        progress_callback=None,
        progress_range=(0.18, 0.30),
    ) -> int:
        os.makedirs(masks_dir, exist_ok=True)
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not image_files:
            print("[-] No keyframes found for masking.")
            return 0

        total = len(image_files)
        start_p, end_p = progress_range
        masked_instances = 0
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        for i, filename in enumerate(image_files):
            img_path = os.path.join(images_dir, filename)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            mask = np.full((h, w), 255, dtype=np.uint8)

            results = self.model.predict(
                frame, conf=self.confidence, device=self.device, verbose=False
            )

            if results and results[0].masks is not None and results[0].boxes is not None:
                result = results[0]
                names = result.names
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                seg_masks = result.masks.data.cpu().numpy()

                for seg_mask, class_id in zip(seg_masks, class_ids):
                    class_name = names.get(class_id, "")
                    if class_name not in DYNAMIC_CLASS_NAMES:
                        continue
                    resized = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    binary_obj = (resized > 0.5).astype(np.uint8) * 255
                    binary_obj = cv2.dilate(binary_obj, dilation_kernel, iterations=1)
                    mask[binary_obj > 0] = 0
                    masked_instances += 1

            out_path = os.path.join(masks_dir, filename + ".png")
            cv2.imwrite(out_path, mask)

            if progress_callback:
                frac = start_p + (end_p - start_p) * ((i + 1) / total)
                progress_callback(frac, f"Masking dynamic objects ({i + 1}/{total})...")

        print(f"[+] Masking complete: {masked_instances} dynamic objects masked across {total} frames.")
        return masked_instances