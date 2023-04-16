import torch
import time
from torchmetrics.aggregation import MeanMetric

import matplotlib as mpl
import matplotlib.pyplot as plt

# Define training loop
def train(loader, model, optimizer, scheduler, loss_fn, metric_fn, device):
    model.train()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()

    # print(f"[training]") # 확인용 코드

    train_loader_cnt = 0
    for inputs, targets in loader:
        # if train_loader_cnt % (len(loader) // 10) == 0:
        #     print(f"[train loader for loop] {train_loader_cnt} / {len(loader)}")  # 확인용 코드

        # # 확인용 코드
        # some_retina = inputs[0]
        # print(some_retina.shape)
        # some_retina_img = some_retina.reshape(224, 224)  # 50,176 -> 224 x 224 변형
        #
        # plt.imshow(some_retina_img, cmap="binary")
        # plt.axis("off")
        #
        # # save_fig("some_digit_plot")
        # plt.show()
        # ###########################################

        start = time.time() # 확인용 코드
        inputs = inputs.to(device)
        targets = targets.to(device)
        end = time.time() # 확인용 코드
        # print(f"[inputs to device, targets to device]")  # 확인용 코드
        # print(f"[inputs to device, targets to device] process_time : {end - start}")  # 확인용 코드

        start = time.time()  # 확인용 코드
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)
        end = time.time() # 확인용 코드
        # print(f"[output model outputs, loss, metric]")  # 확인용 코드
        # print(f"[output model outputs, loss, metric] process_time : {end - start}")  # 확인용 코드

        start = time.time()  # 확인용 코드
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time() # 확인용 코드
        # print(f"[learning update end]")  # 확인용 코드
        # print(f"[learning update end] process_time : {end - start}")  # 확인용 코드

        start = time.time()  # 확인용 코드
        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        scheduler.step()
        end = time.time() # 확인용 코드
        # print(f"[learning mean update end]")  # 확인용 코드
        # print(f"[learning mean update end] process_time : {end - start}")  # 확인용 코드

        train_loader_cnt += 1

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary

# Define evaluation loop
def evaluate(loader, model, loss_fn, metric_fn, device):
    model.eval()
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()

    # print(f"[evaluating]") # 확인용 코드

    evaluate_loader_cnt = 0
    for inputs, targets in loader:
        # if evaluate_loader_cnt % (len(loader) // 10) == 0:
        #     print(f"[evaluate loader for loop] {evaluate_loader_cnt} / {len(loader)}")  # 확인용 코드
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        metric = metric_fn(outputs, targets)

        loss_mean.update(loss.to('cpu'))
        metric_mean.update(metric.to('cpu'))

        evaluate_loader_cnt += 1
    
    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary