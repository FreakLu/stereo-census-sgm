# import cv2
# import numpy as np

# # 读取图像
# left = cv2.imread("selected/1.png", 0)
# right = cv2.imread("selected/0.png", 0)

# print("size:", left.shape)

# num_disp = 192
# h, w = left.shape

# # 创建 SGBM
# stereo = cv2.StereoSGBM_create(
#     minDisparity=0,
#     numDisparities=256,
#     blockSize=7,          
#     P1=8 * 1 * 7 * 7,     
#     P2=32 * 1 * 7 * 7,
#     disp12MaxDiff=1,
#     uniquenessRatio=0,
#     speckleWindowSize=0,
#     speckleRange=0,
#     mode=cv2.STEREO_SGBM_MODE_SGBM
# )



# # 计算视差
# disparity = stereo.compute(left, right).astype(np.float32) / 16.0

# disp_show = np.clip(disparity, 0, num_disp)
# disp_show = (disp_show / num_disp * 255).astype(np.uint8)

# cv2.imshow("Left", left)
# cv2.imshow("Disparity", disp_show)
# cv2.waitKey(0)
#### SAD/BT + SGM OpenCV默认方法
import cv2
import numpy as np

def census_8u(img, r=2):
    """Census transform (uint8 image) -> uint32 census code per pixel.
    r=2 means 5x5 window. Center pixel compares with neighbors."""
    h, w = img.shape
    code = np.zeros((h, w), dtype=np.uint32)
    center = img

    bit = 0
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(img)
            y0, y1 = max(0, dy), min(h, h+dy)
            x0, x1 = max(0, dx), min(w, w+dx)
            shifted[y0:y1, x0:x1] = img[y0-dy:y1-dy, x0-dx:x1-dx]
            code |= ((shifted < center).astype(np.uint32) << bit)
            bit += 1
            if bit >= 32:  # uint32 only
                return code
    return code

def popcount32(x):
    # vectorized popcount for uint32
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    x = (x + (x >> 4)) & np.uint32(0x0F0F0F0F)
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & np.uint32(0x0000003F)

left = cv2.imread("selected/1.png", 0)
right = cv2.imread("selected/0.png", 0)

# 1) Census
cl = census_8u(left, r=2)   # 5x5
cr = census_8u(right, r=2)

# 2) Build cost volume by Hamming distance
min_disp = 0
num_disp = 192
h, w = left.shape
cost = np.full((h, w, num_disp), 63, dtype=np.uint8)

for d in range(num_disp):
    # right shifted to align with left for disparity d
    shifted = np.zeros_like(cr)
    if d < w:
        shifted[:, d:] = cr[:, :w-d]
    hd = popcount32(cl ^ shifted).astype(np.uint8)
    cost[:, :, d] = hd

# cost: (H, W, D) uint8
k = 7  # 可试 5/7/9
cost_agg = np.empty_like(cost)

for d in range(num_disp):
    cost_agg[:, :, d] = cv2.boxFilter(cost[:, :, d], ddepth=-1, ksize=(k, k))

disp = np.argmin(cost_agg, axis=2).astype(np.float32)

# visualize
disp_vis = (disp / num_disp * 255).astype(np.uint8)
cv2.imshow("Left", left)
cv2.imshow("Census-WTA disparity", disp_vis)
cv2.waitKey(0)