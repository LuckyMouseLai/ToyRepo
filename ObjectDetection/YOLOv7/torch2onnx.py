
import torch
import onnx
import os
from models.experimental import attempt_load
device = torch.device('cpu')
modelpath = './checkpoints/yolov7.pt'
# model = create_model(modelpath).to(device)
model = attempt_load(modelpath, map_location=device)
input_names = ['input_0']
output_names= ['output_0']
input = torch.ones(1,3,512,512).to(device)
print(f'Model input names: {input_names}')
print(f'Model output names: {output_names}')
print('Exporting model to ONNX...')
savepath = texts = os.path.splitext(modelpath)[0] + '.onnx'
print(savepath)
torch.onnx.export(model, input, savepath, export_params=True,
                      verbose=True, output_names=output_names,
                      input_names=input_names, opset_version=11, dynamic_axes={'input_0':[2,3], 'output_0':[2,3]})
print('Done...')












