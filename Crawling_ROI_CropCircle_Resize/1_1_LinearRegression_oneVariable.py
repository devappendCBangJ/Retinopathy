import numpy as np

# 학습 데이터
x_train = np.array([21.04, 14.16, 15.34, 8.52, 18.74, 11.34])
y_train = np.array([460., 232., 315., 178., 434., 203.])
n_data = len(x_train)

# 파라미터 초기화
W = np.random.rand()
b = np.random.rand()

# 학습 횟수, 학습률
epochs = 500000
learning_rate = 0.0001

# 학습
for i in range(epochs):
    hypothesis = x_train * W + b
    cost = np.sum((hypothesis - y_train) ** 2) / n_data
    gradient_W = np.sum((W * x_train - y_train + b) * x_train) / n_data
    gradient_b = np.sum((W * x_train - y_train + b)) / n_data

    W -= learning_rate * gradient_W
    b -= learning_rate * gradient_b

    if i % 100 == 0:
        print('Epoch ({:3d}/{:3d}) cost : {:3f}, W : {:3f}, b : {:3f}'.format(i, epochs, cost, W, b))

print('W : {:3f}'.format(W))
print('b : {:3f}'.format(b))
print('result : ')
print(x_train * W + b)