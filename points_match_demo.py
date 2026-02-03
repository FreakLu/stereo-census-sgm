from api.estimate_diaparity_points import estimate_depth_at_points
import cv2

if __name__ == "__main__":

    while True:
        results, vis = estimate_depth_at_points(
            img_stereo_path="/tmp/latest_frame.png",
            yolo_json="/tmp/latest_yolo.json",
            calib_path="calibration.json",
            fx=2692.666,
            baseline=42.11257,
            filter_names=["broadleaf_weed"],
            return_vis=True,
            show_vis=False
        )

        for r in results:
            print(vars(r))

        cv2.imshow("result",vis)
        cv2.waitKey(30)