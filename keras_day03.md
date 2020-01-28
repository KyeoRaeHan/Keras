# Keras_day03

```python
#1-1. 데이터전처리
import numpy as np
x = np.array([range(1, 101), range(101, 201), range(301,401)])
y = np.array([range(101, 201)])

x = np.transpose(x) # (100,3)
y = np.transpose(y)  # (100,1)

#1-2. 데이터셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, random_state=66, shuffle = False)  #데이터를 6:2:2로 나눔, shuffle은 True값
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state=66, shuffle = False) 

# 2. 함수형 API을 이용한 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs = input1, outputs = output1)

model.summary()
```



### Summary
![summary2](https://user-images.githubusercontent.com/59241047/73275572-a7d47600-422a-11ea-91f9-35444b5163c0.JPG)










###  함수형 API

Sequential 클래스와 달리, 함수형 API는 다중입력모델, 다중출력모델, 그래프 구조를 띤 모델 등 유연하고 복잡한 모델을 만들 수 있다.
