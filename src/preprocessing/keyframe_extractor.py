import os
import cv2
import shutil
import numpy as np


class KeyframeExtractor:
    def __init__(self, max_frames=65, min_flow_threshold=1.5):
        self.max_frames = max_frames
        self.min_flow_threshold = min_flow_threshold

    def _clear_directory(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def extract_keyframes(self, video_path: str, output_dir: str):
        print(f"[+] Initializing Smart Keyframe Extractor (target={self.max_frames} frames)...")
        self._clear_directory(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"[-] Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"[-] Video has no readable frames: {video_path}")

        window_size = max(1, total_frames // self.max_frames)
        saved_frames = []
        prev_small_gray = None

        frame_idx = 0
        best_frame_in_window = None
        best_sharpness = -1.0
        best_idx = 0

        print(f"[+] Scanning {total_frames} frames @ {fps:.1f} FPS (window={window_size})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame_in_window = frame.copy()
                best_idx = frame_idx

            if (frame_idx + 1) % window_size == 0 and len(saved_frames) < self.max_frames:
                if best_frame_in_window is not None:
                    small_gray = cv2.resize(
                        cv2.cvtColor(best_frame_in_window, cv2.COLOR_BGR2GRAY), (320, 180)
                    )
                    keep = True
                    if prev_small_gray is not None:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_small_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        mag = float(np.linalg.norm(flow, axis=2).mean())
                        if mag < self.min_flow_threshold:
                            keep = False

                    if keep:
                        filename = f"frame_{len(saved_frames):05d}.jpg"
                        out_path = os.path.join(output_dir, filename)
                        cv2.imwrite(out_path, best_frame_in_window, [cv2.IMWRITE_JPEG_QUALITY, 98])
                        saved_frames.append({
                            "path": out_path,
                            "filename": filename,
                            "frame_idx": best_idx,
                            "timestamp_sec": best_idx / fps,
                        })
                        prev_small_gray = small_gray

                best_frame_in_window = None
                best_sharpness = -1.0

            frame_idx += 1

        cap.release()

        if len(saved_frames) < 5:
            raise RuntimeError(
                f"[-] Only {len(saved_frames)} keyframes had sufficient motion. "
                f"Ensure the drone video has lateral camera translation."
            )

        print(f"[+] Extracted {len(saved_frames)} sharp, parallax-verified keyframes to {output_dir}")
        return saved_frames
