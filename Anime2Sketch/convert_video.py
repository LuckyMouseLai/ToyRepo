import cv2
import time
import argparse
import imageio
import os

from inference import Inference


def read_camera(args):
    camera_number = 0  # 0: 笔记本自带， 1是USB外接, 台式机外接用0
    video_path = args.video_path
    if args.save_path is None:
        texts = os.path.splitext(video_path)
        save_path =texts[0]+'_sketch'+ texts[1]
    else:
        save_path = args.save_path
    input_format = video_path.split('.')[-1]
    if input_format != save_path.split('.')[-1]:
        raise OSError('输入和保存格式不一致！！！')
    if input_format in ['gif', 'GIF']:
        is_gif = True
    elif input_format in ['mp4', 'avi']:
        is_gif = False
    else:
        raise OSError('错误格式！！！')
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():  # 判断视频流是否打开
        print('Failed to read camera')
        exit()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频的平均帧率
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码类型，后缀名.avi
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 编码类型，后缀名.mp4
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    if is_gif:
        gif_frames = []
    else:
        writer = cv2.VideoWriter(save_path,fourcc ,fps, (width, height))
    predictor = Inference(device=args.device, use_onnx=args.use_onnx)
    ## -----
    print('Converting...')
    if is_gif:
        while True:
            flag, frame = cap.read()
            if not flag:
                print('Failed to read frame')
                cap.release()
                break
            new_frame = predictor.inference(frame)  # 模型推理
            gif_frames.append(new_frame)  # 用于gif保存
        imageio.mimsave(save_path, gif_frames, fps=fps)
    else:
        while True:
            flag, frame = cap.read()
            if not flag:
                print('Failed to read frame')
                cap.release()
                writer.release()
                break
            new_frame = predictor.inference(frame)  # 模型推理
            writer.write(new_frame)  # 将帧写入保存视频对象中
    cv2.destroyAllWindows()
    print('Converting Done!')

def get_parser():
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    # parse.add_argument('--mode', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parse.add_argument('--video_path', type=str, default='./images/vinland_saga.gif')
    parse.add_argument('--use_onnx', type=bool, default=False)
    parse.add_argument('--save_path', type=str, default=None, help='video or gif save path, it should be same with input format')
    return parse.parse_args()

if __name__ == '__main__':
    args = get_parser()
    read_camera(args)