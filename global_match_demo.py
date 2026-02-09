from pipeline.dense_depth import estimate_depth_global

if __name__ == "__main__":

    disp = estimate_depth_global(
        img_path="data/images/stereo_01.png",
        calib_path="calibration.json",
        need_rectify = True,
        min_disp=120,num_disp=160,
        P1=8,P2=48,
        show_vis = True
    )