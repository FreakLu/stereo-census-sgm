# 检测引导的目标深度检测算法

本算法适用于由检测模型（如YOLO v11）的输出作为先验条件的目标点深度检测
同时也提供单帧图像全图稠密深度的接口

## 环境安装

虚拟环境

```bash
conda create -n detection_guided_stereo_env python=3.10 -y
conda activate detection_guided_stereo_env
```

安装依赖

```bash
pip install numpy opencv-python numba
```

上游检测模型程序中，推荐使用原子替换保证本算法读入成功

```python
LATEST_IMG = "/tmp/latest_frame.png"
TMP_IMG = "/tmp/latest_frame_tmp.png"
LATEST_JSON = "/tmp/latest_yolo.json"
TMP_JSON = "/tmp/latest_yolo_tmp.json"


def save_latest_frame(frame):
    ok = cv2.imwrite(TMP_IMG, frame)
    os.replace(TMP_IMG, LATEST_IMG)


def save_latest_yolo_json(line_obj):
    with open(TMP_JSON, "w", encoding="utf-8") as f:
        json.dump(line_obj, f, ensure_ascii=False)
    os.replace(TMP_JSON, LATEST_JSON)
```

通过JSON传递点集数据，定义格式如下

```json
{
    "frame":0,
    "detections":[
        {
            "side":"U",
            "x":2855,
            "y":1109,
            "cls":0,
            "name":"broadleaf_weed",
            "conf":0.7416,
        },
        {
            "side":"U",
            "x":671,
            "y":953,
            "cls":0,
            "name":"broadleaf_weed",
            "conf":0.7407,
        }
        //..
    ]
}
```

运行示例参见global_match_demo.py和points_match_demo.py

