import cv2
import numpy as np
from stereo.census import census_transform
from stereo.cost_volume import build_cost_volume_census,build_cost_volume_census_range
from stereo.sgm import sgm_disparity_8dir_numba
from stereo.viz import disp_to_vis_percentile, show_pair
from stereo.rectify import seperation,rectify

def estimate_depth_global(
        img_path:str,
        calib_path:str,
        P1:int = 8,P2:int = 40,
        return_vis:bool=True,
) -> np.ndarray:
    frame = cv2.imread(img_path)
    left,right = seperation(frame)
    left,right = rectify(left,right,calib_path)
    Lr_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    Rr_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    census_L = census_transform(Lr_gray, window_size=2)
    census_R = census_transform(Rr_gray, window_size=2)
    cost_vol = build_cost_volume_census_range(census_L, census_R, min_disp=160,num_disp=120)

    disp = sgm_disparity_8dir_numba(cost_vol, P1=8, P2=48)

    disp_vis = disp_to_vis_percentile(disp, lo_offset=35, hi_offset=5, p_med=50, p_hi=95)
    show_pair(left, disp_vis, scale=0.5)

    return disp