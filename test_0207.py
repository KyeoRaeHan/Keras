import numpy as np
import pandas as pd


# 데이터 불러오기
samsung = np.load('./samsung/data/samsung.npy')
kospi200 = np.load('./samsung/data/kospi200.npy')

def split_xy3(dataset, time_steps, y_column): 
    x, y = list(), list()
    for i in range(len(dataset)): 
        x_end_number = i + time_steps 
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :] 
        tmp_y = dataset[x_end_number:y_end_number,3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy3(samsung, 3, 1) 
x2, y2 = split_xy3(kospi200, 3, 1) 

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, random_state=1, test_size=0.3, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, random_state=1, test_size=0.3, shuffle=False)


# 데이터 전처리
# 3차원 -> 2차원으로으로 변형
x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))


# 데이터 표준화
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()  
scaler.fit(x1_train, x2_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)
x2_train_scaled = scaler.transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)


#2차원 -> 3차원으로 변형
x1_train_scaled = np.reshape(x1_train_scaled,(x1_train_scaled.shape[0], 3,5))
x1_test_scaled = np.reshape(x1_test_scaled,(x1_test_scaled.shape[0], 3,5))
x2_train_scaled = np.reshape(x1_train_scaled,(x1_train_scaled.shape[0], 3,5))
x2_test_scaled = np.reshape(x1_test_scaled,(x1_test_scaled.shape[0], 3,5))


# 모델구성
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense

input1 = Input(shape=(3,5))  
Dense1 = LSTM(8, activation='relu')(input1) 
Dense2 = Dense(32, activation='relu')(Dense1)   
Dense3 = Dense(64, activation='relu')(Dense2)
Dense4 = Dense(8, activation='relu')(Dense3)
output1 = Dense(1)(Dense4)

input2 = Input(shape=(3,5))
Dense12 = LSTM(32, activation='relu')(input2) 
Dense22 = Dense(8, activation='relu')(Dense12)   
Dense23 = Dense(32, activation='relu')(Dense22)
Dense24 = Dense(8, activation='relu')(Dense23)
output2 = Dense(1)(Dense24)

#모델 합치기
from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2]) 

input3 = Dense(4)(merge1)
Dense31 = Dense(16, activation='relu')(input3) 
Dense32 = Dense(8, activation='relu')(Dense31)   
Dense33 = Dense(32, activation='relu')(Dense32)
output3 = Dense(1)(Dense33)

# 모델 정의
model = Model(inputs = [input1, input2],  outputs = output3)

# 모델 훈련
from keras.callbacks import EarlyStopping

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2, epochs= 300, batch_size=1, verbose=2, callbacks=[early_stopping]) 

# 평가&예측
loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1) 
print(loss, mse) 

samsung_2 = np.array([[60100, 61100, 59700, 61100, 14727159], [60000, 60200, 58900, 59500, 19278165],
                    [57100,59000,56800,58900, 21800192], [55500, 57400, 55200, 57200, 23995260]])
kospi200_2 = np.array([[294.74,	300.98,	294.32,	300.65,	110215],[293.57, 294.26, 290.3,	292.02,	102639],
                   [285.19,	291.38,	285.11,	290.68,	105451],[280.17, 286.24, 279.78, 285.05, 111063]])

x3, y3 = split_xy3(samsung_2, 3, 1) 
x4, y4 = split_xy3(kospi200_2, 3, 1)

# 3차원 -> 2차원으로으로 변형
x3 = np.reshape(x3, (x3.shape[0], x3.shape[1] * x3.shape[2]))
x4 = np.reshape(x4, (x4.shape[0], x4.shape[1] * x4.shape[2]))

#데이터 표준화
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()  
scaler.fit(x3, x4)
x3_scaled = scaler.transform(x3)
x4_scaled = scaler.transform(x4)

x3_scaled = np.reshape(x3_scaled,(x3_scaled.shape[0], 3,5))
x4_scaled = np.reshape(x4_scaled,(x4_scaled.shape[0], 3,5))

y_prd = model.predict([x3_scaled, x4_scaled], batch_size=1)

for i in range(1):
    print('종가: ', y3[i:], '/ 예측가: ', y_prd[i])
    
    
# 종가:  [[57200]] / 예측가:  [2.724537e+10]
# 왜 이런 값이 나올까요...
