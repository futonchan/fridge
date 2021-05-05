import time
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data.dataset import Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


# グラフを描画する関数の定義
# def save_training_graph(train_acc,val_acc,train_loss,val_loss):
#     # グラフ全体サイズ大きくする
#     plt.rcParams['figure.figsize'] = (15.0, 15.0)
#     # epoch数の配列numpyで
#     x = np.arange(1,NUM_EPOCHS+1)
#     # Figureの初期化
#     fig = plt.figure()
#     # accuracyグラフ
#     acc_ax = fig.add_subplot(2,2,1,ylim=(0,1),title="model_accuracy",xlabel="epoch",ylabel="accuracy")
#     acc_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     acc_ax.plot(x, train_acc, label="train_acc", marker="o")
#     acc_ax.plot(x, val_acc, label="val_acc", marker="o")
#     # lossグラフ
#     loss_ax = fig.add_subplot(2,2,2,title="model_loss",xlabel="epoch",ylabel="loss")
#     loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     loss_ax.plot(x, train_loss, label="train_loss", marker="o")
#     loss_ax.plot(x, val_loss, label="val_loss", marker="o")
#     # 凡例の表示
#     acc_ax.legend()
#     loss_ax.legend()
#     # プロット カレントディレクトリに保存
#     plt.savefig('figure_crossval{}.png')
#     print("Saved Figure")



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def main():
    DATA_DIR = "./data/veget_google_top30"
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    EFFICIENTNET_MODEL = 'efficientnet-b0'
    IMG_SIZE = EfficientNet.get_image_size(EFFICIENTNET_MODEL) # 224

    # 前処理まとめ
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
    }

    CLASS_NAMES = os.listdir(DATA_DIR)
    CLASS_LENGTH = len(CLASS_NAMES)

    all_set_train = torchvision.datasets.ImageFolder(DATA_DIR, transform=data_transforms["train"])  # your dataset
    all_set_val = torchvision.datasets.ImageFolder(DATA_DIR, transform=data_transforms["val"])  # your dataset
    n = len(all_set_train)  # total number of examples
    n_test = int(0.2 * n)  # take ~20% for test
    image_datasets = {}
    image_datasets["train"]  = torch.utils.data.Subset(dataset=all_set_train, indices=range(n_test, n))  # indicsの20%~100%最後
    image_datasets["val"] = torch.utils.data.Subset(dataset=all_set_val, indices=range(n_test))  # indicsの20%

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} # {'train': 3080, 'val': 769}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dataset_sizes)
    print(device)

    # Classify with EfficientNet
    model = EfficientNet.from_pretrained(EFFICIENTNET_MODEL, num_classes=CLASS_LENGTH)
    print(model._fc)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    NUM_EPOCHS = 25

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    save_path = "./weights/test.pth"
    torch.save(model.state_dict(),save_path) # 推奨される方 重みのみ
    # torch.save(net,save_path) # モデル込で保存 別環境で失敗するかも
    print("Saved model to " + save_path)


# multiProcessing使うときにはこれ書く必要があるらしいよ
if __name__ == "__main__":
    main()