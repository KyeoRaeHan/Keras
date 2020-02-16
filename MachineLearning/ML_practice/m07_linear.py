from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


#1. 데이터
boston = load_boston()

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True)


#2. 모델구성
from sklearn.linear_model import LinearRegression, Ridge, Lasso

model1 = LinearRegression()

# model2 = Ridge()

# model3 = Lasso()

#3. 모델 훈련
model1.fit(x_train,y_train)

#4, 모델평가
Linear = model.score(x_test, y_test) # Accuracy만 반환
print(Linear)
      
#2. 모델 구성
model2 = Ridge()

#3. 훈련
model2.fit(x_train,y_train)

#4. 평가
res = model2.score(x_test,y_test) # Accuracy만 반환
print("Res(Ridge):",res)

#############################################################

#2. 모델 구성
model3 = Lasso()

#3. 훈련
model3.fit(x_train,y_train)

#4. 평가
res = model3.score(x_test,y_test) # Accuracy만 반환
print("Res(Lasso):",res)
