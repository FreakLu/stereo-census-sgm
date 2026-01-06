from pathlib import Path
import cv2

in_dir = Path("data/input")

for p in in_dir.iterdir():
    if p.is_file() and p.suffix.lower() == ".png":
        print(p)
        frame = cv2.imread(str(p))
        H, W = frame.shape[:2]
        mid = W // 2
        left  = frame[:, :mid]
        right = frame[:, mid:]

        base = p.stem
        cv2.imwrite(str(in_dir / f"{base}_L.png"), left)
        cv2.imwrite(str(in_dir / f"{base}_R.png"), right)
