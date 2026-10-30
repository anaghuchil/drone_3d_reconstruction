import os
import cv2


class IlluminationEnhancer:
    @staticmethod
    def apply_clahe_preprocessing(input_dir: str, clahe_output_dir: str) -> str:
        os.makedirs(clahe_output_dir, exist_ok=True)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        valid_extensions = (".jpg", ".jpeg", ".png")
        image_files = sorted([
            f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)
        ])

        count = 0
        for filename in image_files:
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(clahe_output_dir, filename)

            img = cv2.imread(src_path)
            if img is None:
                continue

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

            cv2.imwrite(dst_path, enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
            count += 1

        print(f"[+] Created {count}/{len(image_files)} CLAHE-enhanced keyframes.")
        return clahe_output_dir
