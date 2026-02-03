from pathlib import Path
import cv2
from stereo.anno import AnnPoint
from typing import List, Tuple, Optional
import json

def _batch_split_images():
    in_dir = Path("data/images")

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

def load_points_from_yolo(json_path: str) -> List[AnnPoint]:
    """
    只读取 YOLO 输出格式：
    {
      "frame": int,
      "detections": [
        {"x": float, "y": float, "name": str, ...}
      ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts: List[AnnPoint] = []

    if not isinstance(data, dict):
        return pts

    dets = data.get("detections", [])
    if not isinstance(dets, list):
        return pts

    for d in dets:
        if not isinstance(d, dict):
            continue
        if "x" not in d or "y" not in d:
            continue

        pts.append(
            AnnPoint(
                x=float(d["x"]),
                y=float(d["y"]),
                label=str(d.get("name", ""))  # YOLO 的类别名
            )
        )

    return pts

def split_yolo_points(
    pts: List[AnnPoint],
    img_width: int
) -> Tuple[List[AnnPoint], List[AnnPoint]]:
    """
    根据 x 坐标把 YOLO 点拆成 left_pts / right_pts
    右图点的 x 会减去 img_width，变回右图坐标系
    """
    left_pts: List[AnnPoint] = []
    right_pts: List[AnnPoint] = []

    for p in pts:
        if p.x < img_width:
            left_pts.append(p)
        else:
            right_pts.append(
                AnnPoint(
                    x=p.x - img_width,
                    y=p.y,
                    label=p.label
                )
            )
    return left_pts, right_pts

def filter_points_by_name(
    pts: List[AnnPoint],
    keep_names: List[str]
) -> List[AnnPoint]:
    """
    只保留 label 在 keep_names 里的点
    """
    return [p for p in pts if p.label in keep_names]

def split_stereo_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入：左右合并图 (H, 2W, 3)
    输出：imgL, imgR
    """
    H, W2 = img.shape[:2]
    W = W2 // 2
    imgL = img[:, :W].copy()
    imgR = img[:, W:].copy()
    return imgL, imgR

