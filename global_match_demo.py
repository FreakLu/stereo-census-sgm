from api.estimate_disparity_map import estimate_depth_global

if __name__ == "__main__":

    disp = estimate_depth_global(
        img_path="data/images/stereo_01.png",
        calib_path="calibration.json",
        P1=8,P2=40,
        show_vis = True
    )