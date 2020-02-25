#### 랜덤 포레스트

나무가 모여 숲을 이룬다 = 결정 트리가 앙상블 되어 랜덤 포레스트가 된다



#### 파라미터 

**n_estimators:** 랜덤 포레스트 안의 결정 트리 갯수

n_estimators는 클수록 좋습니다. 결정 트리가 많을수록 더 깔끔한 Decision Boundary가 나오겠죠. 하지만 그만큼 메모리와 훈련 시간이 증가합니다.

**max_features:** 무작위로 선택할 Feature의 개수

max_features=n_features이면 30개의 feature 중 30개의 feature 모두를 선택해 결정 트리를 만듭니다. 단, bootstrap=True이면 30개의 feature에서 복원 추출로 30개를 뽑습니다. 특성 선택의 무작위성이 없어질 뿐 샘플링의 무작위성은 그대로인 것입니다. bootstrap=True는 default 값입니다. 따라서 max_features 값이 크면 랜덤 포레스트의 트리들이 매우 비슷해지고, 가장 두드러진 특성에 맞게 예측을 할 것입니다. max_features 값이 작으면 랜덤 포레스트의 트리들이 서로 매우 달라질 것입니다. 따라서 오버피팅이 줄어들 것입니다. max_features는 일반적으로 Defalut 값을 씁니다.
