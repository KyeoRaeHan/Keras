# Keras_Day01

```python
#1. 데이터(정제된 데이터)
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()  #Sequential:layer순서대로 연산이 이루어지도록 만들어주는 함수  

model.add(Dense(5, input_dim = 1)) 
#input: 1개, hidden layer안 node: 5개, dim =dimension = 차원
model.add(Dense(3)) 
#hidden layer안 node: 3개
model.add(Dense(1)) 
#output: 1개 #input과 output의 갯수는 같아야 한다.


#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x, y, epochs= 100, batch_size=5)

#손실함수의 종류
#평균 제곱 오차(mse:mean squared error)
#평균 절대 오차(mae:mean absolute error)
#평균 제곱근 오차(rmse:root mean square error)
#Rmae: ???

#4.평가예측
loss, mse = model.evaluate(x,y, batch_size=5)
print('mse : ', mse)
```



#### 평균 제곱 오차(mse:mean squared error)

###### 정답에 대한 오류 수치. 낮을수록 정답에 가깝다(좋다)

<img src="C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200122230654149.png" alt="image-20200122230654149" style="zoom:150%;" />

#### 평균 절대 오차 (Mean Absolute Error)

![image-20200122230202536](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200122230202536.png)