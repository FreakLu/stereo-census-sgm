import numpy as np

def popcount_uint32(x:np.ndarray) -> np.ndarray:
    count = np.zeros_like(x, dtype = np.uint8)
    for i in range(64):
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
        if d > 0:
            cost[:, :d] = 255
        cost_volume[:, :, i] = cost

    return cost_volume