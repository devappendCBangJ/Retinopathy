import os
import time
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

from src_VGGNet.models import VGG
from src_VGGNet.models import VGG11
from src_VGGNet.models import VGG13
from src_VGGNet.models import VGG16
from src_VGGNet.models import VGG19
from src_VGGNet.engines import train, evaluate
from src_VGGNet.utils import load_checkpoint, save_checkpoint

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
        "title" : "1_3_baseline_DatasetFolderLoad_Split_VGGNet16_new_regularization",
        "device" : "cuda",
        "root" : "data",
        "batch_size" : 64,
        "num_workers" : 4,
        "epochs" : 100,
        "lr" : 0.001,
        "weight_decay" : 0.0001,
        "label_smoothing" : 0.05,
        "logs" : "logs",
        "checkpoints" : "checkpoints",
        "resume" : False,
        "train_ratio" : 0.7,
        "val_ratio" : 0.15,
        "test_ratio" : 0.15
    })

# Build Dataset
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
            images_path = glob.glob(f'{root}/{category}/*.png')
            for image_route in images_path:
                self.data.append((image_route, label))
    
    # data 개수
    def __len__(self):
        return len(self.data)
    
    # 경로에 있는, 지정한 idx의 이미지 읽기 -> RGB 변환 -> tranform -> image, label 반환
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = self.read_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    # 경로에 있는 image 읽기 -> RGB 변환
    def read_image(self, path):
        image = Image.open(path)
        return image.convert('RGB')

def main(args):
    # Build dataset
    # train, val, test dataset load + make loader
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) # transform을 train과 test 따로따로 설정해줘야하나?
    dataset_root = "data/Retina_Some"
    dataset = RetinaDataset(dataset_root, transform)
    dataset_size = len(dataset)
    train_size = int(dataset_size * args.train_ratio)
    val_size = int(dataset_size * args.val_ratio)
    test_size = dataset_size - train_size - val_size # random_split에서 dataset_size = train_size + val_size + test_size가 되지 않으면 오류 발생
    
    print("[dataset load complete]") # 확인용 코드
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    
    # 확인용 코드
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(val_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")
    
    print(f"train_dataset 개수 : {len(train_dataset)}, " + f"train_loader 개수 : {len(train_loader)}") # train_data 개수, train_loader batch set 개수
    print(f"val_dataset 개수 : {len(val_dataset)}, " + f"val_loader 개수 : {len(val_loader)}") # val_data 개수, val_loader batch set 개수
    print(f"test_dataset 개수 : {len(test_dataset)}, " + f"test_loader 개수 : {len(test_loader)}") # test_data 개수, test_loader batch set 개수

    # Build model
    model = VGG16()
    model = model.to(args.device)
    
    # Build optimizer
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Build scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))

    # Build loss function
    loss_fn = nn.CrossEntropyLoss()
    """
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    """

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
        # start timer
        start_time = time.time() # 확인용 코드

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

        # stop timer
        end_time = time.time() # 확인용 코드
        
        # Print log
        print((
            f'[Epoch {epoch + 1}] '
            + f'{epoch + 1}epoch time {end_time - start_time:.02f}, '
            + f'Train Loss {train_summary["loss"]:.04f}, '
            + f'Train Accuracy {train_summary["metric"]:.04f}, '
            + f'Test Loss {val_summary["loss"]:.04f}, '
            + f'Test Accuracy {val_summary["metric"]:.04f}'
        ))

if __name__=="__main__":
    main(args)
