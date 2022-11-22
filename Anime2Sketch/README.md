# Anime2Sketch
# 1. 模型
参考论文和Github，以及torch模型来源：[Anima2Sketch](https://engineering.purdue.edu/people/xiaoyu.xiang.1) [Github](https://github.com/Mukosame/Anime2Sketch)<br/>
将模型放于checkpoints/目录下，torch和onnx模型:[百度云](https://pan.baidu.com/s/1mD1ORh4TfWZsjcpZCoFDlg)
**提取码：xyca**<br/>

# 2. 环境配置
1. 在win10 + torch1.8.0 + python3.7环境下测试
2. 安装依赖: pip install -r requirements.txt

# 3. 效果展示
<img src="./images/abcd.gif" alt="Original Input"><img src="./images/abcd_sketch.gif" alt="Original Input">

# 4. 代码说明
1. torch2onnx.py -- 将模型转为onnx模型
    python torch2onnx.py
    - 将在原模型下导出相同名称的onnx模型
<br/>

2. convert_image.py -- 图片级别的测试
    python convert_image.py --device 'cpu' --path [your_img_path] --use_onnx False --save_path [your_save_path]
    - device: 使用cpu或cuda，默认cpu
    - path: 输入图片的路径, 同时也是结果图片的输出目录
    - use_onnx: 是否使用onnx模型，默认False
    - save_path: 保存路径，默认保存输入图片路径下
<br/>

3. convert_video.py -- 视频转换，非实时
    python convert_video.py --device 'cpu' --video_path [your_video_path] --use_onnx False --save_path [your_save_path]
    - device: 使用cpu或cuda，默认cpu
    - video_path: 输入视频的路径, 同时也是结果图片的输出目录
    - use_onnx: 是否使用onnx模型，默认False
    - save_path: 保存路径，默认保存输入视频路径下
<br/>

4. gradio demo
    - gradio_image_demo.py : 图片级别转换可视化demo
    - gradio_video_demo.py : 视频级别转换可视化demo
