import os
import glob
import argparse
import easydict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional.classification import accuracy

from PIL import Image

from src.models import ConvNet
from src.engines import train, evaluate
from src.utils import load_checkpoint, save_checkpoint

# # Jupyter 외 환경
# parser = argparse.ArgumentParser()
# parser.add_argument("--title", type=str, default="baseline")
# parser.add_argument("--device", type=str, default="cuda")
# parser.add_argument("--root", type=str, default="data")
# parser.add_argument("--batch_size", type=int, default=64)
# parser.add_argument("--num_workers", type=int, default=2)
# parser.add_argument("--epochs", type=int, default=100)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--logs", type=str, default='logs')
# parser.add_argument("--checkpoints", type=str, default='checkpoints')
# parser.add_argument("--resume", type=bool, default=False)
# args = parser.parse_args()

# Jupyter 환경
args = easydict.EasyDict({
        "title" : "baseline",
        "device" : "cuda",
        "root" : "data",
        "batch_size" : 64,
        "num_workers" : 2,
        "epochs" : 100,
        "lr" : 0.001,
        "logs" : "logs",
        "checkpoints" : "checkpoints",
        "resume" : False
    })

class RetinaDataset(Dataset):
    # image dataset 전체 경로 저장 -> tranform
    def __init__(self, root, transform=None):
        super(RetinaDataset, self).__init__()
        self.make_dataset(root)
        self.transform = transform
    
    # image dataset 전체 경로 저장
    def make_dataset(self, root):
        # class(폴더명) 불러오기
        self.data = []
        categories = os.listdir(root)
        categories = sorted(categories)
        
        # class -> label 변환 + 각 class의 이미지 파일 전부 가져오기
        for label, category in enumerate(categories):
            images = glob.glob(f'{root}/{category}/*.png')
            for image in images:
                self.data.append((image, label))
    
    # data 개수
    def __len__(self):
        return len(self.data)
    
    # 경로에 있는, 지정한 idx의 이미지 읽기 -> RGB 변환 -> tranform -> image, label 반환
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.read_image(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    # 경로에 있는 image 읽기 -> RGB 변환
    def read_image(self, path):
        image = Image.open(path)
        return image.convert('RGB')

def main(args):
    # Build dataset
    # - load train dataset + make loader
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_root = 'data/Cifar10/train'
    train_data = RetinaDataset(train_root, train_transform)
    # train_data = CIFAR10(args.root, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # - load val dataset + make loader
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_root = 'data/Cifar10/test' # test!!! val!!!
    val_data = RetinaDataset(val_root, val_transform)
    # val_data = CIFAR10(args.root, train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # 확인용 코드
    print(f"train_data 개수 : {len(train_data)}, " + f"train_loader 개수 : {len(train_loader)}") # train_data 개수, train_loader batch set 개수
    print(f"val_data 개수 : {len(val_data)}, " + f"val_loader 개수 : {len(val_loader)}") # train_data 개수, train_loader batch set 개수

    # Build model
    model = ConvNet()
    model = model.to(args.device)

    # Build optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Build scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    # Build loss function
    loss_fn = nn.CrossEntropyLoss()

    # Build metric function
    metric_fn = accuracy

    # Build logger
    train_logger = SummaryWriter(f'{args.logs}/train/{args.title}')
    val_logger = SummaryWriter(f'{args.logs}/val/{args.title}')

    # Load model
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.checkpoints, args.title, model, optimizer)
    
    # Main loop
    for epoch in range(start_epoch, args.epochs):
        # train one epoch
        train_summary = train(train_loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
        
        # evaluate one epoch
        val_summary = evaluate(val_loader, model, loss_fn, metric_fn, args.device)

        # write log
        train_logger.add_scalar('Loss', train_summary['loss'], epoch + 1)
        train_logger.add_scalar('Accuracy', train_summary['metric'], epoch + 1)
        val_logger.add_scalar('Loss', val_summary['loss'], epoch + 1)
        val_logger.add_scalar('Accuracy', val_summary['metric'], epoch + 1)

        # save model
        save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)
        
        # Print log
        print((
            f'Epoch {epoch + 1}: '
            + f'Train Loss {train_summary["loss"]:.04f}, '
            + f'Train Accuracy {train_summary["metric"]:.04f}, '
            + f'Test Loss {val_summary["loss"]:.04f}, '
            + f'Test Accuracy {val_summary["metric"]:.04f}'
        ))
        
        break # 테스트●●●

if __name__=="__main__":
    main(args)