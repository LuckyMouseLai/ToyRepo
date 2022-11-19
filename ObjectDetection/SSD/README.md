# SSD ObjectDetection
# 1. 模型
根据此仓库提供的代码和模型编写的一个SSD目标检测demo, 训练代码也请查看该仓库：[lufficc-SSD](https://github.com/lufficc/SSD)<br/>
将模型放于ObjectDetection/SSD/checkpoints/目录下，模型:[百度云](https://pan.baidu.com/s/1mD1ORh4TfWZsjcpZCoFDlg)
**提取码：xyca**<br/>



# 2. 环境配置
1. 在win10 + torch1.8.0 + python3.6环境下测试
2. 安装依赖: pip install -r requirements.txt

# 3. 检测类别
    - VOC: class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')
    - coco: class_names = ('__background__',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush')
# 4. 代码说明

1. pred_image.py -- 图片级别的目标检测
    python .\pred_image.py --model_number 0 --device 'cpu' --score_threshold 0.5 --image_path './images/003123.jpg'
    - model_number: 使用的模型, [0,1,2,3]，默认0。需要将相应的权重文件放到./checkpoints文件夹下
        - 0：backbone-VGG image size-300 train on: voc
        - 1：backbone-VGG image size-512 train on: voc
        - 2：backbone-VGG image size-300 train on: coco
        - 3：backbone-VGG image size-512 train on: coco
    - device: 使用cpu或cuda，默认cpu
    - score_threshold: 检测目标的概率阈值，只有概率大于阈值的目标才算成功检测。默认设置0.5
    - image_path: 输入图片的路径, 同时也是结果图片的输出目录
    - 可根据具体需要调整阈值，检测时注意把image_path修改为你检测的图片路径
    python .\pred_image.py --model_number 0 --score_threshold 0.5 --image_path 'your image path'
<br/>    

2. pred_video.py -- 实时目标检测，调用摄像头检测
    python pred_video.py --model_number 0 --device 'cuda' --mode 0 
    - model_number: 使用的模型, [0,1,2,3]，默认0。需要将相应的权重文件放到./checkpoints文件夹下
        - 0：backbone-VGG image size-300 train on: voc  class_number: 21
        - 1：backbone-VGG image size-512 train on: voc  class_number: 21
        - 2：backbone-VGG image size-300 train on: coco  class_number: 81
        - 3：backbone-VGG image size-512 train on: coco  class_number: 81
    - device: 使用cpu或cuda，默认cuda
    - score_threshold: 检测目标的概率阈值，只有概率大于阈值的目标才算成功检测。默认设置0.2
    - 实时检测粗略测试
        - vgg_ssd300_voc0712.pth模型实时结果：3060显卡实时fps 26 fps
        - vgg_ssd512_voc0712.pth模型实时结果：3060显卡实时fps 19 fps
        - vgg_ssd300_coco_trainval35k.pth模型实时结果：3060显卡实时fps 25 fps
        - vgg_ssd512_coco_trainval35k.pth模型实时结果：3060显卡实时fps 15 fps
        - 前三个检测效果较差，很多东西检测不到，第4个好一些，可能检测环境跟训练数据集相差比较大, 调整阈值可能会有更好的结果, 4个实时不会有明显的卡顿。
<br/>

3. gradio_video_demo.py -- gradio可视化部署
    - 运行可在浏览器可视化应用查看demo
    - 输入为视频，将结果保存为temp.mp4在该目录下
    ```
    python gradio_video_demo.py
    ```







