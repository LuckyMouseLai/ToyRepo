# YOLOv7
# 1. 模型
根据[YOLOv7 paper](https://arxiv.org/abs/2207.02696)的官方提供的代码[YOLOv7](https://github.com/WongKinYiu/yolov7)和预训练模型编写的一个目标检测demo <br/>
将模型放于ObjectDetection/YOLO-v7/checkpoints/目录下，模型:[百度云](https://pan.baidu.com/s/1mD1ORh4TfWZsjcpZCoFDlg)
**提取码：xyca**<br/>



# 2. 环境配置
1. 在win10 + torch1.8.0 + python3.6环境下测试
2. 安装依赖: pip install -r requirements.txt

# 3. 检测类别
- class = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


# 4. 代码说明

1. detect_image.py -- 图片级别的目标检测
    python .\detect_image.py --image_path './test.png' --device '0'
    - device: 使用的设备，cpu或cuda。注意: 支持'cpu, 0, 1, 2...'，使用cpu则传入'cpu', 使用cuda则传入设备编号, 如'0'。使用cuda时模型使用半精度推理。默认'0'
    - conf_threshold: 检测目标的概率阈值(置信度阈值)，只有概率大于阈值的目标才算成功检测。默认设置0.25
    - iou_threshold: 非极大值抑制时的iou阈值。默认设置0.45
    - image_path: 输入图片的路径, 同时也是结果图片的输出目录
    - image_size: 推理时的图片大小，默认640, 需根据模型决定, 这里使用官方提供的yolov7.pt权重，支持640大小图片。
    - 可根据具体需要调整置信度阈值，检测时注意把image_path修改为你检测的图片路径
    python .\detect_image.py --image_path 'your image path'
<br/>    

2. detect_video.py -- 实时目标检测，调用摄像头检测
    python .\detect_video.py --device '0' 
    - device: 使用的设备，cpu或cuda。注意: 支持'cpu, 0, 1, 2...'，使用cpu则传入'cpu', 使用cuda则传入设备编号, 如'0'。使用cuda时模型使用半精度推理。默认'0'
    - conf_threshold: 检测目标的概率阈值(置信度阈值)，只有概率大于阈值的目标才算成功检测。默认设置0.25
    - iou_threshold: 非极大值抑制时的iou阈值。默认设置0.45
    - image_size: 推理时的图片大小，默认640, 需根据模型决定, 这里使用官方提供的yolov7.pt权重，支持640大小图片。
    - 实时检测粗略测试
        - yolov7.pt模型实时结果：3060显卡-实时fps 22 fps
        - yolov7.pt模型实时结果：cpu-不支持实时
<br/>







