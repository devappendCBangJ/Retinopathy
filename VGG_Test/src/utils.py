import os
import torch

def load_checkpoint(checkpoint_dir, title, model, optimizer):
    # state_dict 불러오기(epoch, model, optimizer)
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    state_dict = torch.load(checkpoint_path)
    start_epoch = state_dict['epoch']
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return start_epoch

def save_checkpoint(checkpoint_dir, title, model, optimizer, epoch):
    # state_dict 저장(epoch, model, optimizer)
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    torch.save(state_dict, checkpoint_path)
    
def save_transform(transform_dir, train_transforms, test_transforms, title):
    # transform 정보 저장
    os.makedirs(transform_dir, exist_ok=True)
    transform_txt = open(f'{transform_dir}/{title}.txt', 'a')
    transform_txt.write("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡtrain_transformsㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n")
    transform_txt.write(f"{str(train_transforms)}\n\n")
    transform_txt.write("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡtest_transformsㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n")
    transform_txt.write(f"{str(test_transforms)}\n")
    transform_txt.close()

def save_best_param(transform_dir, train_transforms, test_transforms, lr, title):
    # transform 정보 저장
    os.makedirs(transform_dir, exist_ok=True)
    transform_txt = open(f'{transform_dir}/{title}_best_transform.txt', 'w')
    transform_txt.write("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡtrain_transformsㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n")
    transform_txt.write(f"{str(train_transforms)}\n\n")
    transform_txt.write("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡtest_transformsㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n")
    transform_txt.write(f"{str(test_transforms)}\n\n")
    transform_txt.write("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡlearning_rateㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n")
    transform_txt.write(f"{str(lr)}\n\n")
    transform_txt.close()