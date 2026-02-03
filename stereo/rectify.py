import json
import numpy as np
import cv2

def load_stereo_json(json_path: str):
    with open(json_path, "r") as f:
        j = json.load(f)

    K1 = np.array(j["K1"], dtype=np.float64)
    D1 = np.array(j["D1"][0], dtype=np.float64).reshape(-1, 1)   # (5,1)

    K2 = np.array(j["K2"], dtype=np.float64)
    D2 = np.array(j["D2"][0], dtype=np.float64).reshape(-1, 1)

    R = np.array(j["R"], dtype=np.float64)
    T = np.array(j["T"], dtype=np.float64).reshape(3, 1)         # 单位：mm

    w, h = j["image_size"]
    return K1, D1, K2, D2, R, T, (w, h)

def build_rectify_maps(K1, D1, K2, D2, R, T, size, alpha=0):
    w, h = size

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2,
        (w, h),
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, (w, h), cv2.CV_16SC2
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, (w, h), cv2.CV_16SC2
    )
    return {
        "map1x": map1x, "map1y": map1y,
        "map2x": map2x, "map2y": map2y,
        "R1": R1, "R2": R2,
        "P1": P1, "P2": P2,
        "Q": Q,
        "size": (w, h),
    }

def rectify_pair(left_bgr, right_bgr, maps):
    left_rect  = cv2.remap(left_bgr,  maps["map1x"], maps["map1y"], cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, maps["map2x"], maps["map2y"], cv2.INTER_LINEAR)
    return left_rect, right_rect

def rectify(left, right, json_path: str,return_maps=False):
    K1, D1, K2, D2, R, T, size = load_stereo_json(json_path)
    maps = build_rectify_maps(K1, D1, K2, D2, R, T, size, alpha=0)
    left_rect, rigth_rect = rectify_pair(left, right, maps)
    if return_maps:
        return left_rect, rigth_rect, maps
    return left_rect,rigth_rect

def rectify_points(points, K, D, R_rect, P_rect):
    """
    points:  (N,2) array-like or list of (x,y)
    returns: (N,2) float32 in rectified image pixel coords
    """
    if points is None:
        return None
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)  # (N,1,2)
    out = cv2.undistortPoints(pts, K, D, R=R_rect, P=P_rect)      # (N,1,2)
    out = out.reshape(-1, 2).astype(np.float32)                   # (N,2)
    return out

def rectify_points_lr(left_points, right_points, json_path: str, alpha=0):
    """
    left_points/right_points: (N,2) or list of (x,y)
    返回:left_points_rect, right_points_rect, maps
    """
    K1, D1, K2, D2, R, T, size = load_stereo_json(json_path)
    maps = build_rectify_maps(K1, D1, K2, D2, R, T, size, alpha=alpha)

    left_rect_pts  = rectify_points(left_points,  K1, D1, maps["R1"], maps["P1"])
    right_rect_pts = rectify_points(right_points, K2, D2, maps["R2"], maps["P2"])
    return left_rect_pts, right_rect_pts, maps

def seperation(frame:np.ndarray):
    _, W = frame.shape[:2]
    mid = W // 2
    left_half  = frame[:, :mid]
    right_half = frame[:, mid:]
    return left_half,right_half

def unificaton(left:np.ndarray,right:np.ndarray):
    if left.shape[0] != right.shape[0]:
        raise ValueError("left/right height mismatch")
    return np.hstack([left,right])