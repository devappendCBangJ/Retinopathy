import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from torchmetrics.aggregation import MeanMetric
import torch.nn.functional as F

# Define training loop
def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    # 모델 학습
    model.train()
    
    # 변수 초기화
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()

    train_loader_cnt = 0
    for inputs, targets in loader:
        # input, target 불러오기
        inputs = inputs.to(device)
        targets = targets.to(device)

        # loss, acc 계산
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        # loss, optimizer 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        # scheduler 업데이트
        scheduler.step()

        train_loader_cnt += 1

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary

# Define evaluation loop
def evaluate(loader, model, loss_fn, metric_fn, device):
    # 모델 평가
    model.eval()
    
    # 변수 초기화
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    # print(loss_mean) # 확인용 코드

    evaluate_loader_cnt = 0
    for inputs, targets in loader:
        # input, target 불러오기
        inputs = inputs.to(device)
        targets = targets.to(device)

        # loss, acc 계산
        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)
        # print(loss) # 확인용 코드

        # loss, acc 업데이트
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        evaluate_loader_cnt += 1
    
    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}
    # print(summary) # 확인용 코드
    
    return summary

# 예측한 값 추출 + 맞춘 이미지 정보 추출 + 틀린 이미지 정보 추출
def get_predictions(loader, model, device):
    # 모델 평가
    model.eval()
    
    # 변수 평가
    images = []
    targets = []
    outputs_probs = []

    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            # batch 단위 input 불러오기
            batch_inputs = batch_inputs.to(device)

            # batch 단위 모델 예측 + 예측 확률 계산
            batch_outputs = model(batch_inputs)
            batch_outputs_probs = F.softmax(batch_outputs, dim = -1)
            batch_max_probs_example_idx = batch_outputs_probs.argmax(1, keepdim = True) # 각 열에서 가장 큰 행 index 찾음. keepdim = True : 출력 텐서를 입력과 동일한 크기로 유지
            # print("batch_max_probs_example_idx : ", batch_max_probs_example_idx) # 확인용 코드
            batch_min_probs_example_idx = batch_outputs_probs.argmin(1, keepdim = True) # 각 열에서 가장 작은 행 index 찾음. keepdim = True : 출력 텐서를 입력과 동일한 크기로 유지
            # print("batch_min_probs_example_idx : ", batch_min_probs_example_idx) # 확인용 코드
            
            # batch 단위 이미지 + 실제값 + 예측 확률 저장
            images.append(batch_inputs.cpu())
            targets.append(batch_targets.cpu())
            outputs_probs.append(batch_outputs_probs.cpu())
    
    # 전체 이미지 + 실제값 + 예측 확률 저장(합치기)
    images = torch.cat(images, dim = 0) # 행 방향으로 합치기
    # print("images : ", images) # 확인용 코드
    targets = torch.cat(targets, dim = 0) # 행 방향으로 합치기
    # print("targets : ", targets) # 확인용 코드
    outputs_probs = torch.cat(outputs_probs, dim = 0) # 행 방향으로 합치기
    # print("outputs_probs : ", outputs_probs) # 확인용 코드
    
    # 모델 예측(가장 높은 예측 확률인 값으로 예측)
    outputs = torch.argmax(outputs_probs, 1) # 각 행에서 가장 큰 열 index 값 찾음
    
    # 정답 비교 + 정답 채점
    corrects = torch.eq(targets, outputs)
    
    # 변수 초기화
    correct_examples = []
    wrong_examples = []

    # 맞춘 이미지 + 틀린 이미지 정보 저장
    for image, target, output_probs, correct in zip(images, targets, outputs_probs, corrects):
        if correct==True:
            correct_examples.append((image, target, output_probs))
        elif correct==False:
            wrong_examples.append((image, target, output_probs))
    
    # 맞춘 이미지 + 틀린 이미지 정보 확률순으로 정렬
    correct_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values) # reverse=True : 내림차순 정렬, key : key값을 활용한 정렬(기본값은 오름차순), torch.max(x, dim=0) : 각 행에서 최댓값을 가져온다 -> 여기서는 틀린값 중에서 output_probs가 가장 높은 값을 가져옴(output이 틀린 정답에 가장 확신)
    # print("correct_examples : ", correct_examples)
    wrong_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values) # reverse=True : 내림차순 정렬, key : key값을 활용한 정렬(기본값은 내림차순), torch.max(x, dim=0) : 각 행에서 최댓값을 가져온다 -> 여기서는 틀린값 중에서output_probs가 가장 높은 값을 가져옴(output이 틀린 정답에 가장 확신)
    # print("wrong_examples : ", wrong_examples)
    
    return correct_examples, wrong_examples

def plot_most_correct_wrong(correct_examples, wrong_examples, classes, n_images, normalize=True):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    
    # plot 사이즈 지정
    fig = plt.figure(figsize = (25, 20))
    for i in range(rows * cols):
        # subplot 생성
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # 이미지 + 실제값 + 예측 확률 저장
        image, target, output_probs = correct_examples[i]
        image = image.permute(1, 2, 0) # ???
        
        # 각 example의 target 예측 확률 + 최대 예측 확률 추출
        target_prob = output_probs[target]
        output_prob, output = torch.max(output_probs, dim = 0) # 각 열에서 가장 큰 행 원소값 찾음
        target_class = classes[target]
        output_class = classes[output]
        
        # 이미지 출력을 위한 정규화
        if normalize:
            image = normalize_image(image)
        
        # 이미지 출력
        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label : {target_class} ({target_prob:.3f})\n' \
                     f'pred label : {output_class} ({output_prob:.3f})')
        ax.axis('off')
    
    fig.subplots_adjust(hspace = 0.4) # ???
    
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    
    # plot 사이즈 지정
    fig = plt.figure(figsize = (25, 20))
    for i in range(rows * cols):
        # subplot 생성
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # 이미지 + 실제값 + 예측 확률 저장
        image, target, output_probs = wrong_examples[i]
        image = image.permute(1, 2, 0) # ???
        
        # 각 example의 target 예측 확률 + 최대 예측 확률 추출
        target_prob = output_probs[target]
        output_prob, output = torch.max(output_probs, dim = 0) # 각 열에서 가장 큰 행 원소값 찾음
        target_class = classes[target]
        output_class = classes[output]
        
        # 이미지 출력을 위한 정규화
        if normalize:
            image = normalize_image(image)
        
        # 이미지 출력
        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label : {target_class} ({target_prob:.3f})\n' \
                     f'pred label : {output_class} ({output_prob:.3f})')
        ax.axis('off')
    
    fig.subplots_adjust(hspace = 0.4) # ???

# 모델 학습 소요시간
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs

# 이미지 출력을 위한 정규화
def normalize_image(image):
    # 이미지 최소값, 최대값 추출
    image_min = image.min()
    image_max = image.max()
    
    # 이미지 최소값, 최대값 지정 : 설정 최소값 이하 -> 설정 최소값으로 변경, 설정 최대값 이상 -> 설정 최대값으로 변경
    image.clamp_(min = image_min, max = image_max) # ???
    image.add_(-image_min).div_(image_max - image_min + 1e-5) # ???
    
    return image