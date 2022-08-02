import torch
import onnx
import os
from network import BiSeNet

device = torch.device('cpu')
modelpath = '../../checkpoints/79999_iter.pth'
model = BiSeNet(n_classes=19).to(device)
model.load_state_dict(torch.load(modelpath), strict=False)
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
                      input_names=input_names, opset_version=11)
print('Done...')