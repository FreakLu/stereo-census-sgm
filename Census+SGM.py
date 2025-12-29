import numpy as np
import cv2
import time
from numba import njit
import json


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
    return map1x, map1y, map2x, map2y, Q

def rectify_pair(left_bgr, right_bgr, maps):
    map1x, map1y, map2x, map2y, _ = maps
    left_rect  = cv2.remap(left_bgr,  map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_bgr, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect

INF = 1 << 29  # 足够大

@njit(cache=True)
def dp_update_1d(prev_L, c, P1, P2, out_L):
    D = prev_L.shape[0]

    # min_prev
    min_prev = prev_L[0]
    for d in range(1, D):
        v = prev_L[d]
        if v < min_prev:
            min_prev = v

    base_jump = min_prev + P2

    for d in range(D):
        best = prev_L[d]

        if d > 0:
            v = prev_L[d-1] + P1
            if v < best:
                best = v

        if d < D - 1:
            v = prev_L[d+1] + P1
            if v < best:
                best = v

        if base_jump < best:
            best = base_jump

        out_L[d] = c[d] + best - min_prev


@njit(cache=True)
def sgm_aggregate_8dir_numba(cost_vol, P1, P2):
    """
    cost_vol: (H, W, D)
    return S: (H, W, D) int32
    8 dirs: → ← ↓ ↑  ↘ ↖ ↗ ↙
    """
    H, W, D = cost_vol.shape
    C = cost_vol.astype(np.int32)
    S = np.zeros((H, W, D), dtype=np.int32)

    L_prev = np.zeros(D, dtype=np.int32)
    L_cur  = np.zeros(D, dtype=np.int32)

    # ---------- 1) left -> right (→) ----------
    for y in range(H):
        for d in range(D): L_prev[d] = 0
        for x in range(W):
            if x == 0:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

    # ---------- 2) right -> left (←) ----------
    for y in range(H):
        for d in range(D): L_prev[d] = 0
        for x in range(W-1, -1, -1):
            if x == W-1:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

    # ---------- 3) top -> bottom (↓) ----------
    for x in range(W):
        for d in range(D): L_prev[d] = 0
        for y in range(H):
            if y == 0:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

    # ---------- 4) bottom -> top (↑) ----------
    for x in range(W):
        for d in range(D): L_prev[d] = 0
        for y in range(H-1, -1, -1):
            if y == H-1:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

    # ---------- 5) top-left -> bottom-right (↘) ----------
    # starts on left edge (y0,0)
    for y0 in range(H):
        for d in range(D): L_prev[d] = 0
        y = y0
        x = 0
        first = True
        while y < H and x < W:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y += 1
            x += 1

    # starts on top edge (0,x0), x0>=1
    for x0 in range(1, W):
        for d in range(D): L_prev[d] = 0
        y = 0
        x = x0
        first = True
        while y < H and x < W:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y += 1
            x += 1

    # ---------- 6) bottom-right -> top-left (↖) ----------
    # starts on right edge (y0,W-1), y0 from H-1..0
    for y0 in range(H-1, -1, -1):
        for d in range(D): L_prev[d] = 0
        y = y0
        x = W - 1
        first = True
        while y >= 0 and x >= 0:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y -= 1
            x -= 1

    # starts on bottom edge (H-1,x0), x0<=W-2
    for x0 in range(W-2, -1, -1):
        for d in range(D): L_prev[d] = 0
        y = H - 1
        x = x0
        first = True
        while y >= 0 and x >= 0:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y -= 1
            x -= 1

    # ---------- 7) bottom-left -> top-right (↗) ----------
    # starts on left edge (y0,0), y0 from H-1..0
    for y0 in range(H-1, -1, -1):
        for d in range(D): L_prev[d] = 0
        y = y0
        x = 0
        first = True
        while y >= 0 and x < W:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y -= 1
            x += 1

    # starts on bottom edge (H-1,x0), x0>=1
    for x0 in range(1, W):
        for d in range(D): L_prev[d] = 0
        y = H - 1
        x = x0
        first = True
        while y >= 0 and x < W:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y -= 1
            x += 1

    # ---------- 8) top-right -> bottom-left (↙) ----------
    # starts on right edge (y0,W-1), y0 from 0..H-1
    for y0 in range(H):
        for d in range(D): L_prev[d] = 0
        y = y0
        x = W - 1
        first = True
        while y < H and x >= 0:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y += 1
            x -= 1

    # starts on top edge (0,x0), x0<=W-2
    for x0 in range(W-2, -1, -1):
        for d in range(D): L_prev[d] = 0
        y = 0
        x = x0
        first = True
        while y < H and x >= 0:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y += 1
            x -= 1

    return S


def sgm_disparity_8dir_numba(cost_vol, P1=8, P2=32):
    S = sgm_aggregate_8dir_numba(cost_vol, P1, P2)
    disp = np.argmin(S, axis=2).astype(np.uint16)
    return disp

def census_transform(img: np.ndarray,window_size: int = 2) ->np.ndarry:
    assert img.ndim ==2, "grayscale only"
    H,W = img.shape
    code = np.zeros((H,W),dtype = np.uint32)

    center = img.astype(np.uint32)

    bit = 0
    for dy in range(-window_size,window_size+1):
        for dx in range(-window_size,window_size+1):
            if dy == 0 and dx == 0:
                continue

            shifted = np.zeros((H,W),dtype = np.uint32)

            y0 = max(0,dy)
            y1 = min(H,H+dy)
            x0 = max(0,dx)
            x1 = min(W,W+dx)

            shifted[y0:y1,x0:x1] = center[y0-dy:y1-dy,x0-dx:x1-dx]

            b = (shifted < center).astype(np.uint32)

            code |= (b<<bit)

            bit += 1

            if bit > 48:
                break
        if bit > 48:
            break
    
    return code

def popcount_uint32(x:np.ndarray) -> np.ndarray:
    count = np.zeros_like(x, dtype = np.uint8)
    for i in range(32):
        count += (x>>i)&1
    return count

def build_cost_volume_census(census_L,census_R,max_disp):
    assert census_L.shape == census_R.shape, "input error"
    H,W = census_L.shape

    cost_volume = np.zeros((H,W,max_disp),dtype = np.uint8)

    for d in range(max_disp):
        shifted_R = np.zeros_like(census_R)

        if d > 0:
            shifted_R[:,d:] = census_R[:,:-d]
        else:
            shifted_R = census_R.copy()
        
        xor = census_L ^ shifted_R
        cost = popcount_uint32(xor)
        if d > 0:
            cost[:, :d] = 255   # 这 d 列是无效匹配，强制惩罚
        cost_volume[:,:,d] = cost

    return cost_volume

def build_cost_volume_census_range(census_L, census_R, min_disp, num_disp):
    assert census_L.shape == census_R.shape, "input error"
    H, W = census_L.shape

    cost_volume = np.zeros((H, W, num_disp), dtype=np.uint8)

    for i in range(num_disp):
        d = min_disp + i

        shifted_R = np.zeros_like(census_R)

        # 只考虑正视差（正常左图为参考）
        if d > 0:
            shifted_R[:, d:] = census_R[:, :-d]
        else:
            shifted_R = census_R.copy()

        xor = census_L ^ shifted_R
        cost = popcount_uint32(xor)
        cost_volume[:, :, i] = cost

    return cost_volume

INF = 1 << 30

def sgm_aggregate_8dir(cost_vol: np.ndarray,
                       P1: int = 2,
                       P2: int = 24) -> np.ndarray:
    """
    cost_vol: (H, W, D)
    return S: (H, W, D), int32 aggregated cost
    8 directions: → ← ↓ ↑  ↘ ↖ ↗ ↙
    """
    assert cost_vol.ndim == 3
    H, W, D = cost_vol.shape
    C = cost_vol.astype(np.int32)
    S = np.zeros((H, W, D), dtype=np.int32)

    def dp_update(prev_L: np.ndarray, c: np.ndarray) -> np.ndarray:
        min_prev = prev_L.min()

        prev_minus = np.empty_like(prev_L)
        prev_plus  = np.empty_like(prev_L)

        prev_minus[0] = INF
        prev_minus[1:] = prev_L[:-1]

        prev_plus[-1] = INF
        prev_plus[:-1] = prev_L[1:]

        t0 = prev_L
        t1 = prev_minus + P1
        t2 = prev_plus + P1
        t3 = min_prev + P2

        m = np.minimum(t0, np.minimum(t1, np.minimum(t2, t3)))
        return c + m - min_prev

    # -------- 1) left -> right (→) --------
    for y in range(H):
        L_prev = np.zeros(D, np.int32)
        for x in range(W):
            L = C[y, x].copy() if x == 0 else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L

    # -------- 2) right -> left (←) --------
    for y in range(H):
        L_prev = np.zeros(D, np.int32)
        for x in range(W - 1, -1, -1):
            L = C[y, x].copy() if x == W - 1 else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L

    # -------- 3) top -> bottom (↓) --------
    for x in range(W):
        L_prev = np.zeros(D, np.int32)
        for y in range(H):
            L = C[y, x].copy() if y == 0 else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L

    # -------- 4) bottom -> top (↑) --------
    for x in range(W):
        L_prev = np.zeros(D, np.int32)
        for y in range(H - 1, -1, -1):
            L = C[y, x].copy() if y == H - 1 else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L

    # -------- 5) top-left -> bottom-right (↘) --------
    # scan order: y increasing, x increasing
    for y0 in range(H):
        y, x = y0, 0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y < H and x < W:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y += 1; x += 1
    for x0 in range(1, W):
        y, x = 0, x0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y < H and x < W:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y += 1; x += 1

    # -------- 6) bottom-right -> top-left (↖) --------
    # reverse of ↘ : y decreasing, x decreasing
    for y0 in range(H - 1, -1, -1):
        y, x = y0, W - 1
        L_prev = np.zeros(D, np.int32)
        first = True
        while y >= 0 and x >= 0:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y -= 1; x -= 1
    for x0 in range(W - 2, -1, -1):
        y, x = H - 1, x0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y >= 0 and x >= 0:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y -= 1; x -= 1

    # -------- 7) bottom-left -> top-right (↗) --------
    # scan order: y decreasing, x increasing
    for y0 in range(H - 1, -1, -1):
        y, x = y0, 0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y >= 0 and x < W:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y -= 1; x += 1
    for x0 in range(1, W):
        y, x = H - 1, x0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y >= 0 and x < W:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y -= 1; x += 1

    # -------- 8) top-right -> bottom-left (↙) --------
    # reverse of ↗ : y increasing, x decreasing
    for y0 in range(H):
        y, x = y0, W - 1
        L_prev = np.zeros(D, np.int32)
        first = True
        while y < H and x >= 0:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y += 1; x -= 1
    for x0 in range(W - 2, -1, -1):
        y, x = 0, x0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y < H and x >= 0:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y += 1; x -= 1

    return S

def sgm_disparity(cost_vol: np.ndarray,img: np.ndarray, P1: int = 8, P2: int = 32) -> np.ndarray:
    S = sgm_aggregate_8dir(cost_vol,P1=P1, P2=P2)
    disp = np.argmin(S, axis=2).astype(np.uint16)
    return disp


# ------------------
json_path = "calibration.json"

frame = cv2.imread("selected/image.png")
H, W = frame.shape[:2]
mid = W // 2
left_bgr  = frame[:, :mid]
right_bgr = frame[:, mid:]

K1, D1, K2, D2, R, T, size = load_stereo_json(json_path)
maps = build_rectify_maps(K1, D1, K2, D2, R, T, size, alpha=0)

Lr, Rr = rectify_pair(left_bgr, right_bgr, maps)

left = cv2.imread("selected/1.png", 0)
right = cv2.imread("selected/0.png", 0)

Lr_gray = cv2.cvtColor(Lr, cv2.COLOR_BGR2GRAY)
Rr_gray = cv2.cvtColor(Rr, cv2.COLOR_BGR2GRAY)

Census_L = census_transform(Lr_gray,2)
Census_R = census_transform(Rr_gray,2)

min_disp = 190
num_disp = 80

volume = build_cost_volume_census_range(Census_L,Census_R,min_disp, num_disp)

print(volume[200,300,:])

t0 = time.time()

disp_wta = np.argmin(volume,axis=2)
disp_sgm = sgm_disparity_8dir_numba(volume,P1=4, P2=36)
disp_sgm = disp_sgm + min_disp
print("SGM time:", time.time() - t0)

disp = disp_sgm.astype(np.int32)
D = 384  # 你当前的 D

print("min/max:", disp.min(), disp.max())
print("p1/p5/p50/p95/p99:", np.percentile(disp, [1,5,50,95,99]))

print("ratio == 0:", np.mean(disp == 0))
print("ratio == D-1:", np.mean(disp == (D-1)))

disp = disp_sgm.astype(np.float32)

med = np.percentile(disp, 50)
hi  = np.percentile(disp, 95)

# 窗口：从 (med - 40) 到 (hi + 10)，你可以调 40/10
lo = max(0, med - 35)
hi = hi + 5

disp_vis = np.clip((disp - lo) / (hi - lo + 1e-6), 0, 1)
disp_vis = (disp_vis * 255).astype(np.uint8)



scale = 0.5  # 0.5=缩小一半
disp_show = cv2.resize(disp_vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
left_show = cv2.resize(Lr_gray,  None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

cv2.imshow("left", left_show)
cv2.imshow("disp", disp_show)
#cv2.imshow("wta",disp_wta)
cv2.waitKey(0)
