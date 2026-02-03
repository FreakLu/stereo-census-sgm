import numpy as np

def extract_patch(img: np.ndarray, cx: int, cy: int, r: int, pad_mode: str = "reflect") -> np.ndarray:
    """
    从 img 里裁一个以 (cx, cy) 为中心，半径 r 的方形 patch，大小 (2r+1, 2r+1)
    若越界，则用 np.pad 按 pad_mode 填充后再裁。
    img: 2D 灰度图 (H, W)
    """
    assert img.ndim == 2, "grayscale only"
    H, W = img.shape
    ps = 2 * r + 1

    # 需要 padding 的量
    pad = r

    # 先 pad，再把坐标平移 pad
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode=pad_mode)

    cx2 = cx + pad
    cy2 = cy + pad

    patch = padded[cy2 - r: cy2 + r + 1, cx2 - r: cx2 + r + 1]
    assert patch.shape == (ps, ps), f"patch shape wrong: {patch.shape}"
    return patch