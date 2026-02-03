from api.estimate_diaparity_points import estimate_depth_at_points

if __name__ == "__main__":

    results, vis = estimate_depth_at_points(
        img_stereo_path="/tmp/latest_frame.png",
        yolo_json="/tmp/latest_yolo.json",
        calib_path="calibration.json",
        fx=2692.666,
        baseline=42.11257,
        filter_names=["broadleaf_weed"],
        return_vis=True
    )

    for r in results:
        print(vars(r))