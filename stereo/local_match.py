import cv2
import numpy as np

from stereo.anno import AnnPoint
from typing import List, Tuple, Optional

def build_candidates(
    left_pts: List[AnnPoint],
    right_pts: List[AnnPoint],
    dy: float = 6.0,
    dmin: float = 0.0,
    dmax: float = 400.0
) -> List[List[int]]:
    """
    candidates[i] = list of right indices j that satisfy:
      |yL - yR| <= dy
      dmin <= (xL - xR) <= dmax
    """
    cand: List[List[int]] = []
    for p in left_pts:
        ci: List[int] = []
        for j, q in enumerate(right_pts):
            if abs(p.y - q.y) > dy:
                continue
            d = p.x - q.x
            if d < dmin or d > dmax:
                continue
            ci.append(j)
        cand.append(ci)
    return cand


# -----------------------------
# 3) 统计：看 gating 松紧是否合理
# -----------------------------
def print_candidate_stats(candidates: List[List[int]], name: str = "stats"):
    lens = np.array([len(c) for c in candidates], dtype=np.int32)
    if len(lens) == 0:
        print(f"[{name}] no left points")
        return
    print(f"[{name}] left_count={len(lens)}")
    print(f"[{name}] cand_mean={float(lens.mean()):.3f}  cand_median={float(np.median(lens)):.3f}  cand_max={int(lens.max())}")
    print(f"[{name}] ratio_empty={float(np.mean(lens == 0)):.3f}  ratio_gt10={float(np.mean(lens > 10)):.3f}")


# -----------------------------
# 4) 可视化：拼接左右图 + 点 + 候选连线
# -----------------------------
def to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("image is None (check path)")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def draw_step1_vis(
    imgL: np.ndarray,
    imgR: np.ndarray,
    left_pts: List[AnnPoint],
    right_pts: List[AnnPoint],
    candidates: List[List[int]],
    scale: float = 0.7,
    max_lines_per_left: int = 30,
    show_index: bool = True
) -> np.ndarray:
    imgL = to_bgr(imgL)
    imgR = to_bgr(imgR)

    H1, W1 = imgL.shape[:2]
    H2, W2 = imgR.shape[:2]
    if H1 != H2:
        # 简单处理：把右图 resize 到同高（只为可视化，真实匹配建议用矫正后同尺寸）
        imgR = cv2.resize(imgR, (int(W2 * H1 / H2), H1), interpolation=cv2.INTER_AREA)
        H2, W2 = imgR.shape[:2]

    canvas = np.hstack([imgL, imgR])
    offset_x = W1

    # 右点：绿圈
    for j, q in enumerate(right_pts):
        cv2.circle(canvas, (int(q.x + offset_x), int(q.y)), 5, (0, 255, 0), 2)
        if show_index:
            cv2.putText(canvas, f"R{j}", (int(q.x + offset_x) + 6, int(q.y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 左点：红圈 + 候选线
    for i, p in enumerate(left_pts):
        xL, yL = int(p.x), int(p.y)
        cv2.circle(canvas, (xL, yL), 5, (0, 0, 255), 2)
        if show_index:
            cv2.putText(canvas, f"L{i}", (xL + 6, yL - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        js = candidates[i]
        if len(js) > max_lines_per_left:
            js = js[:max_lines_per_left]

        for j in js:
            q = right_pts[j]
            xR, yR = int(q.x + offset_x), int(q.y)
            # 候选线：浅黄蓝（看得清但不刺眼）
            cv2.line(canvas, (xL, yL), (xR, yR), (255, 200, 0), 1)

    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas


# ===== Step2: Census-based point matching (local refinement) =====
from stereo.census import census_transform

# 8-bit popcount lookup table (no .bit_count dependency)
_POPCNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def _patch_bbox(x: int, y: int, r: int, W: int, H: int):
    x0 = x - r
    x1 = x + r + 1
    y0 = y - r
    y1 = y + r + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    return x0, x1, y0, y1

def _hamming_cost_patch_u32(patchL_u32: np.ndarray, patchR_u32: np.ndarray) -> int:
    """
    patchL_u32, patchR_u32: (h,w) uint32 census codes
    cost = sum popcount( patchL XOR patchR ) over all pixels in patch
    """
    xor = np.bitwise_xor(patchL_u32, patchR_u32)  # uint32
    # view as bytes and popcount with LUT
    xb = xor.view(np.uint8)  # shape (h,w,4) on little-endian
    return int(_POPCNT_LUT[xb].sum())

def census_match_one_candidate(
    census_L: np.ndarray, census_R: np.ndarray,
    xL: int, yL: int,
    xR: int, yR: int,
    r: int = 8,
    dx_max: int = 5,
    dy_max: int = 0
):
    """
    For a fixed candidate right point (xR,yR), do local search around it:
      dx in [-dx_max, +dx_max], dy in [-dy_max, +dy_max]
    Return:
      best_cost, best_dx, best_dy, second_cost
    """
    H, W = census_L.shape
    bbL = _patch_bbox(xL, yL, r, W, H)
    if bbL is None:
        return None  # left patch out of bounds

    x0L, x1L, y0L, y1L = bbL
    patchL = census_L[y0L:y1L, x0L:x1L]  # (2r+1,2r+1)

    best_cost = 1 << 60
    second_cost = 1 << 60
    best_dx = 0
    best_dy = 0

    # small local search
    for dy in range(-dy_max, dy_max + 1):
        yy = yR + dy
        for dx in range(-dx_max, dx_max + 1):
            xx = xR + dx
            bbR = _patch_bbox(xx, yy, r, W, H)
            if bbR is None:
                continue
            x0R, x1R, y0R, y1R = bbR
            patchR = census_R[y0R:y1R, x0R:x1R]

            c = _hamming_cost_patch_u32(patchL, patchR)

            if c < best_cost:
                second_cost = best_cost
                best_cost = c
                best_dx = dx
                best_dy = dy
            elif c < second_cost:
                second_cost = c

    if best_cost >= (1 << 59):  # no valid right patch
        return None

    return best_cost, best_dx, best_dy, second_cost

def step2_match_points_census(
    imgL_gray: np.ndarray, imgR_gray: np.ndarray,
    left_pts: List[AnnPoint], right_pts: List[AnnPoint],
    candidates: List[List[int]],
    census_window_size: int = 2,
    patch_r: int = 8,        # 17x17
    dx_max: int = 5,
    dy_max: int = 0,
    # thresholds (you WILL tune these)
    max_best_cost: int = 999999999,
    min_gap: int = 0
):
    """
    Returns a list of match results aligned to left_pts:
      matches[i] = dict or None
    dict fields:
      j, xR_ref, yR_ref, disp, best_cost, second_cost, gap, dx, dy
    """
    # 1) census on full rectified images (reuse validated module)
    census_L = census_transform(imgL_gray, window_size=census_window_size)
    census_R = census_transform(imgR_gray, window_size=census_window_size)

    matches = [None] * len(left_pts)

    for i, p in enumerate(left_pts):
        js = candidates[i]
        if not js:
            continue

        xL, yL = int(round(p.x)), int(round(p.y))

        best_overall = None  # (best_cost, j, best_dx, best_dy, second_cost)

        # evaluate each candidate right point
        for j in js:
            q = right_pts[j]
            xR, yR = int(round(q.x)), int(round(q.y))

            res = census_match_one_candidate(
                census_L, census_R,
                xL, yL,
                xR, yR,
                r=patch_r,
                dx_max=dx_max,
                dy_max=dy_max
            )
            if res is None:
                continue

            best_cost, best_dx, best_dy, second_cost = res
            if best_overall is None or best_cost < best_overall[0]:
                best_overall = (best_cost, j, best_dx, best_dy, second_cost)

        if best_overall is None:
            continue

        best_cost, j_star, dx_star, dy_star, second_cost = best_overall
        gap = int(second_cost - best_cost) if second_cost < (1 << 59) else 999999999

        # reject rules (tune later)
        if best_cost > max_best_cost:
            continue
        if gap < min_gap:
            continue

        q = right_pts[j_star]
        xR0, yR0 = int(round(q.x)), int(round(q.y))
        xR_ref = xR0 + int(dx_star)
        yR_ref = yR0 + int(dy_star)

        disp = (xL - xR_ref)

        matches[i] = {
            "j": j_star,
            "xR_ref": xR_ref,
            "yR_ref": yR_ref,
            "disp": disp,
            "best_cost": int(best_cost),
            "second_cost": int(second_cost if second_cost < (1 << 59) else -1),
            "gap": int(gap),
            "dx": int(dx_star),
            "dy": int(dy_star),
        }

    return matches

def draw_step2_final_matches(
    imgL: np.ndarray, imgR: np.ndarray,
    left_pts: List[AnnPoint], right_pts: List[AnnPoint],
    matches: List[Optional[dict]],
    scale: float = 0.7,
    show_index: bool = True
) -> np.ndarray:
    imgL = to_bgr(imgL)
    imgR = to_bgr(imgR)
    H, W = imgL.shape[:2]
    canvas = np.hstack([imgL, imgR])
    off = W

    # draw all points
    for i, p in enumerate(left_pts):
        cv2.circle(canvas, (int(p.x), int(p.y)), 6, (0, 0, 255), 2)
        if show_index:
            cv2.putText(canvas, f"L{i}", (int(p.x) + 8, int(p.y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for j, q in enumerate(right_pts):
        cv2.circle(canvas, (int(q.x + off), int(q.y)), 6, (0, 255, 0), 2)
        if show_index:
            cv2.putText(canvas, f"R{j}", (int(q.x + off) + 8, int(q.y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # draw final 1-to-1 matches only
    for i, m in enumerate(matches):
        if m is None:
            continue
        p = left_pts[i]
        xL, yL = int(round(p.x)), int(round(p.y))
        xR_ref = int(m["xR_ref"])
        yR_ref = int(m["yR_ref"])

        # thick cyan line for final match
        cv2.line(canvas, (xL, yL), (xR_ref + off, yR_ref), (255, 255, 0), 2)

        # annotate disparity and cost
        txt = f"d={m['disp']} c={m['best_cost']} g={m['gap']}"
        cv2.putText(canvas, txt, (xL + 8, yL + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # mark refined right point center (small cross)
        cv2.drawMarker(canvas, (xR_ref + off, yR_ref), (255, 255, 0),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)

    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas
