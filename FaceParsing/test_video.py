import cv2
import time
import argparse

from inference import Inference

def read_camera(args):
    camera_number = 0  # 0: 笔记本自带， 1是USB外接, 台式机外接用0
    cap = cv2.VideoCapture(camera_number)
    if not cap.isOpened():  # 判断视频流是否打开
        print('Failed to read camera')
        exit()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频的平均帧率
    predictor = Inference(device=args.device, use_onnx=args.use_onnx)
    start_time = time.time()
    count = 0  # 帧数计数
    while True:
        flag, frame = cap.read()
        if not flag:
            print('Failed to read frame')
            cap.release()
            break
        count += 1
        # print(frame.shape)
        new_frame = predictor.inference(frame, args.mode)  # 模型推理
        # print(new_frame.shape)
        cv2.putText(new_frame, "FPS: {0}".format(float('%.1f' % (count / (time.time() - start_time)))), (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.namedWindow('window', cv2.WINDOW_FREERATIO)  # 设置自适应窗口
        cv2.resizeWindow("window", 960, 720)  # 设置显示窗口大小
        cv2.imshow('window', new_frame)  # 可以通过resize frame的大小，达到修改窗口大小的效果
        key = cv2.waitKey(5)  # 检测按键时间，如果按键，返回键值ASCII否则返回-1。如果参数为0，直到按键才继续
        if key == ord('q'):
            cap.release()
            break
    cv2.destroyAllWindows()

def get_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    parse.add_argument('--mode', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parse.add_argument('--use_onnx', type=bool, default=False)
    return parse.parse_args()

if __name__ == '__main__':
    args = get_parser()
    read_camera(args)