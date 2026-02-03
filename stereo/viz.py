import numpy as np
import cv2

def disp_stats(disp: np.ndarray, D: int):
    disp = disp.astype(np.float32)
    print("min/max:", disp.min(), disp.max())
    print("p1/p5/p50/p95/p99:", np.percentile(disp, [1,5,50,95,99]))
    print("ratio == 0:", np.mean(disp == 0))
    print("ratio == D-1:", np.mean(disp == (D-1)))

def disp_to_vis_percentile(disp: np.ndarray,lo_offset=35, hi_offset=5,p_med=50, p_hi=95):
    disp = disp.astype(np.float32)
    med = np.percentile(disp, p_med)
    hi  = np.percentile(disp, p_hi)

    lo = max(0.0, med - lo_offset)
    hi = hi + hi_offset

    vis = np.clip((disp - lo) / (hi - lo + 1e-6), 0, 1)
    return (vis * 255).astype(np.uint8)

def show_pair(left_gray: np.ndarray, disp_vis: np.ndarray, scale=0.5):
    left_show = cv2.resize(left_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    disp_show = cv2.resize(disp_vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("left", left_show)
    cv2.imshow("disp", disp_show)
    cv2.waitKey(0)

def disp_to_vis_linear(disp: np.ndarray, D: int, invalid_zero: bool = True) -> np.ndarray:
    d = disp.astype(np.float32)

    vis = (d / max(1, (D - 1))) * 255.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    return vis