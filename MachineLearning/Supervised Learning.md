머신러닝은 학습하려는 **문제의 유형**에 따라 **지도 학습, 비지도 학습, 강화 학습**으로 나눌 수 있다.
그리고 각각의 학습 방법들은 **상황에 맞는** 다양한 알고리즘을 사용하여 구현할 수 있습니다.

## 지도학습(Supervised Learning)

**레이블된(정답이 정해져있는) 데이터**로 모델을 학습시켜, 이전에 본 적이 없는 새로운 데이터를 **예측**하는 것

*여기서 **지도**는 희망하는 출력신호(레이블)가 있는 샘플들을 의미한다.

***레이블(label)**은 데이터에 정해진 특징을 뜻한다.


#### 분류(Classification): 클래스 레이블 예측

##### 이진분류(Binary Classification)
![binary_class](https://user-images.githubusercontent.com/59241047/74606261-55dc8d00-5112-11ea-825c-f8cef6953ad5.png)

예측하는 데이터의 클래스 레이블이 **2가지**인 경우 해당된다.
ex) 스팸메일, 시험 PASS/FAIL


##### 다중분류(Multiclass Classification)
![multi_classification](https://user-images.githubusercontent.com/59241047/74606390-65a8a100-5113-11ea-8fd0-13117d48adf5.JPG)

예측하는 데이터의 클래스 레이블이 **3개 이상**인 경우 해당된다.

ex) MNIST


#### 회귀(Regression)
<img width="385" alt="linear_regression" src="https://user-images.githubusercontent.com/59241047/74606406-79540780-5113-11ea-802c-340d08a8e82b.png">

연속적인 출력값을 예측하는 것. 
회귀분석은 X(예측변수, 설명변수, 입력)와 Y(반응변수,출력,타깃)가 주어졌을 때,  출력값을 예측하는 두 변수 사이의 관계를 찾습니다.






