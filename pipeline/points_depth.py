import cv2
import numpy as np
import json
import time
from typing import List, Tuple, Optional
from stereo.rectify import rectify_pair,rectify_points_lr
from stereo.anno import AnnPoint,PointDepthResult
from stereo.input import split_stereo_image,split_yolo_points,load_points_from_yolo,filter_points_by_name
from stereo.local_match import build_candidates,print_candidate_stats,step2_match_points_census,draw_step2_final_matches

def estimate_depth_at_points(
    img_stereo_path: str,
    yolo_json: str,
    calib_path: str,
    fx: float, baseline: float,
    filter_names: Optional[List[str]] = None,
    dy: float = 20, dmin: float = 0, dmax: float = 500,
    return_vis: bool = False,
    show_vis: bool = True
) -> Tuple[List[PointDepthResult], Optional[np.ndarray]]:
    """
    Estimate depth at sparse object points detected by YOLO using stereo matching.

    This function takes a stereo image pair (concatenated left-right image) and
    object center points from YOLO detection, performs stereo rectification,
    point-level correspondence search, and computes depth for matched points.

    Args:
        img_stereo_path (str): Path to the concatenated stereo image (left | right).
        yolo_json (str): Path to YOLO detection result in JSON format.
        calib_path (str): Path to stereo camera calibration file.
        fx (float): Focal length in pixels (x direction).
        baseline (float): Stereo baseline in the same unit as depth output.
        filter_names (Optional[List[str]]): Class names to keep; others are ignored.
        dy (float): Vertical tolerance for epipolar constraint (in pixels).
        dmin (float): Minimum allowed disparity.
        dmax (float): Maximum allowed disparity.
        return_vis (bool): Whether to return visualization image.
        show_vis (bool): Whether to display visualization using OpenCV.

    Returns:
        results (List[PointDepthResult]): Depth estimation results for matched points.
        vis (Optional[np.ndarray]): Visualization image if return_vis is True; otherwise None.
    """
    t0 = time.perf_counter()
    img_yolo = cv2.imread(img_stereo_path,cv2.IMREAD_COLOR)
    imgL,imgR = split_stereo_image(img_yolo)

    _,img_width = img_yolo.shape[:2]

    if imgL is None or imgR is None:
        raise RuntimeError("Image load failed")
    
    points = load_points_from_yolo(yolo_json)
    points = filter_points_by_name(points, filter_names)
    #print(points)
    left_pts , right_pts = split_yolo_points(points,img_width/2)
    print("Loaded points:", "L =", len(left_pts), "R =", len(right_pts))
    left_xy  = [(p.x, p.y) for p in left_pts]
    right_xy = [(p.x, p.y) for p in right_pts]

    left_xy, right_xy, maps = rectify_points_lr(
        left_xy, right_xy, calib_path, alpha=0
    )
    imgL, imgR = rectify_pair(imgL, imgR, maps)
    for i, (x, y) in enumerate(left_xy):
        left_pts[i].x = float(x)
        left_pts[i].y = float(y)

    for i, (x, y) in enumerate(right_xy):
        right_pts[i].x = float(x)
        right_pts[i].y = float(y)

    candidates = build_candidates(
        left_pts, right_pts,
        dy=dy, dmin=dmin, dmax=dmax
    )
    print_candidate_stats(
        candidates,
        name=f"dy={dy}, d=[{dmin},{dmax}]"
    )
    L_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    R_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    matches = step2_match_points_census(
        L_gray, R_gray,
        left_pts, right_pts,
        candidates,
        census_window_size=2,
        patch_r=8,
        dx_max=5,
        dy_max=0,
        max_best_cost=999999999,
        min_gap=0
    )

    results: List[PointDepthResult] = []

    for i, m in enumerate(matches):
        if m is None:
            continue
        pL = left_pts[i]
        disp = m["disp"]
        if disp <= 0:
            continue

        depth = fx * baseline / disp

        results.append(
            PointDepthResult(
                label=pL.label,
                x_left=pL.x,
                y_left=pL.y,
                x_right=m["xR_ref"],
                y_right=m["yR_ref"],
                disparity=disp,
                depth=depth,
                cost=m["best_cost"],
                gap=m["gap"],
            )
        )
    vis2 = draw_step2_final_matches(imgL, imgR, left_pts, right_pts, matches, scale=0.5, show_index=True)
    t1 = time.perf_counter()
    print(f"[estimate_depth_at_points] time = {(t1 - t0)*1000:.2f} ms")
    if show_vis:
        cv2.imshow("step2_final_matches", vis2)
        cv2.waitKey(30)
    if return_vis:
        return results, vis2
    else:
        return results, None


    