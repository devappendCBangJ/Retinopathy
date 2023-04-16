import numpy as np

# 학습 데이터
x1 = np.array([73., 93., 89., 96., 73.])
x2 = np.array([80., 88., 91., 98., 66.])
x3 = np.array([75., 93., 90., 100., 70.])
y = np.array([152., 185., 180., 196., 142.])
n_data = len(x1)

# 파라미터 초기화
w1 = np.random.rand()
w2 = np.random.rand()
w3 = np.random.rand()
b = np.random.rand()

print("w1 = ", w1, "w2 = ", w2, "w3 = ", w3, "b = ", b)

# 학습횟수, 학습률
epochs = 5000
learning_rate = 0.00001

# 학습
for i in range(epochs):
    hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
    cost = np.sum((hypothesis - y) ** 2) / n_data
    gradient_w1 = np.sum((x1 * w1 + x2 * w2 + x3 * w3 + b - y) * x1) / n_data
    gradient_w2 = np.sum((x1 * w1 + x2 * w2 + x3 * w3 + b - y) * x2) / n_data
    gradient_w3 = np.sum((x1 * w1 + x2 * w2 + x3 * w3 + b - y) * x3) / n_data
    gradient_b = np.sum((x1 * w1 + x2 * w2 + x3 * w3 + b - y)) / n_data

    w1 -= learning_rate * gradient_w1
    w2 -= learning_rate * gradient_w2
    w3 -= learning_rate * gradient_w3
    b -= learning_rate * gradient_b

    if i % 2 == 0:
        print('Epoch ({:3d}/{:3d}) cost : {:3f}, w1 : {:3f}, w2 : {:3f}, w3 : {:3f}, b : {:3f}'.format(i, epochs, cost, w1, w2, w3, b))

print('w1 : {:3f}, w2 : {:3f}, w3 : {:3f}'.format(w1, w2, w3))
print('b : {:3f}'.format(b))
print('result : ')
print(x1 * w1 + x2 * w2 + x3 * w3 + b)