# train.pyからテスト部分のみを抜き出したファイル
import os
import datetime
from PIL import Image
import copy
import json

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # matplotlib横軸を整数で表示する用
import numpy as np
from torchvision import transforms


from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from sklearn.metrics import precision_score,recall_score,confusion_matrix
import pandas as pd
from sklearn.utils.multiclass import unique_labels

# CPU,GPU両対応
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')


model_name = 'efficientnet-b0'
image_size = EfficientNet.get_image_size(model_name) # 224
# Classify with EfficientNet
model = EfficientNet.from_pretrained(model_name, num_classes=2)

save_path = "test.pth"
pretrained_model = torch.load(save_path, map_location=device)
# print(model)
model.load_state_dict(pretrained_model)
model.to(device)

model.eval()
img = Image.open('./data/hymenoptera_data/val/ants/8124241_36b290d372.jpg')
tfms = transforms.Compose([transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
img = tfms(img).unsqueeze(0)
with torch.no_grad():
    img = img.to(device)
    logits = model(img)
preds = torch.topk(logits, k=2).indices.squeeze(0).tolist()

# Load class names
labels_map = json.load(open("ants_bee_labels.txt"))
labels_map = [labels_map[str(i)] for i in range(2)]
print(labels_map)
print('-----')

for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(label, prob*100))