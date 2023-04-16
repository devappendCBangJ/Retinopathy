# 사이킷런 예제 : https://engineer-mole.tistory.com/16

from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# 숫자 데이터셋 불러오기
digits = datasets.load_digits()

plt.matshow(digits.images[0], cmap = "Greys")
plt.show()

# 데이터셋 분류
x = digits.data
y = digits.target
x_train, y_train = x[0::2], y[0::2]
x_test, y_test = x[1::2], y[1::2]

# 모델 세팅
clf = svm.SVC(gamma=0.001)

# 모델에 데이터셋 학습
clf.fit(x_train, y_train)

# 모델 평가
accuracy = clf.score(x_test, y_test)
print(f"{accuracy}")

predicted = clf.predict(x_test)
print("classification report")
print(metrics.classification_report(y_test, predicted))