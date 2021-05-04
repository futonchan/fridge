import os

import torch
import torchvision
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data.dataset import Subset

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

# multiProcessing使うときにはこれ書く必要があるらしいよ
if __name__ == "__main__":
    main()