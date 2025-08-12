# 방학스터디 3주차

## Predictive Maintenance with XGBoost

### XGBoost

XGBoost는 RandomForest와 유사하게 결정 트리를 사용하지만, 작동 방식에서 차이가 있다. RandomForest에서는 각각의 트리들이 각각 예측한 결과를 바탕으로 순도를 측정해서 질문을 수정해가는 과정을 걸쳐 최종적으로 모든 트리들의 의사를 종합한 예측을 내놓는다.<br>
반면, XGBoost와 같은 Boosting 기법에서는 첫 번째 트리가 예측한 값과 실제 값의 오차를 바탕으로 두 번째, 세 번쨰... 마지막 트리는 오차를 예측하도록 한다.<br>
예를들어, 학생 A, B, C가 10개의 예측정비 문제를 푼다고 가정해보자.<Br>

|데이터 번호|실제 정답|A 예측|오차 (실제 - 예측)|
|:----:|:----:|:----:|:----:|
|1|1 (고장)	    |0.6|	+0.4|
|2|	0 (정상)	|0.3|	-0.3|
|3|	0 (정상)	|0.6|	-0.6|
|4|	1 (고장)	|0.2|	+0.8|
|...|...|...|...|

B는 데이터와 A의 오차를 바탕으로 학습을 수행하는데, 실제 정답을 예측하는 것이 아닌, 오차를 예측하도록 학습한다. A의 예측과 B의 예측 오차를 더한 값이 정답에 대한 예측값이 된다.

|데이터 번호|실제 정답|B 예측 오차|A예측 + B예측|오차|
|:----:|:----:|:----:|:----:|:----:|
|1| 1 (고장)    |0.3|	0.9|    +0.1|
|2|	0 (정상)	|-0.15|	0.15|   -0.15|
|3|	0 (정상)	|-0.5|	0.1|    -0.1|
|4|	1 (고장)	|0.68|	0.88|   +0.12|
|...|...|...|...|

A예측에 B의 오차 예측을 더한 결과와 실제 정답의 오차는 줄어드는 것을 볼 수 있다.<br>
C는 A와 B의 예측 오차를 바탕으로 또 오차를 학습한다. 결과적으로 예측값과 예측 오차를 더한 값은 실제 정답 값에 수렴하게된다.<br>
실제 XGBoost에서는 학습률로 다음 노드들의 오차 예측값을 일정 부분 반영해 안정적으로 정답을 찾아가도록 한다.<br>

## Prediction

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- 1. 데이터 로드 ---
df = pd.read_csv('ai4i2020.csv')

# --- 2. 피처 엔지니어링 ---
print("--- 피처 엔지니어링 수행 ---")
# Power 특성 (Torque * Rotational Speed)
df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
# Temperature Difference 특성
df['Temp_Diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
# Overstrain 특성 (Tool wear * Torque)
df['Overstrain'] = df['Tool wear [min]'] * df['Torque [Nm]']
print("Power, Temp_Diff, Overstrain 특성 3개 추가 완료")


# --- 3. 특성(X)과 타겟(y) 분리 ---
# 식별자, 타겟, 세부 고장 원인 컬럼들 제외
X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
# 'Type' 컬럼 원-핫 인코딩
X = pd.get_dummies(X, columns=['Type'], drop_first=True)


# --- 4. 데이터 분할 ---
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)


# --- 5. XGBoost 모델 생성 및 학습 ---
# 클래스 불균형 처리를 위한 가중치 계산
scale_pos_weight_value = np.sum(y_train == 0) / np.sum(y_train == 1)

model_xgb = xgb.XGBClassifier(objective='binary:logistic',
                              scale_pos_weight=scale_pos_weight_value,
                              use_label_encoder=False,
                              eval_metric='logloss',
                              random_state=42)

print("\n--- XGBoost 모델 학습 시작 (피처 엔지니어링 적용) ---")
model_xgb.fit(X_train, y_train)
print("--- 모델 학습 완료 ---")


# --- 6. 모델 평가 ---
print("\n--- 모델 평가 (테스트 데이터) ---")
y_pred_xgb = model_xgb.predict(X_test)
print("\n--- 분류 리포트 (XGBoost + 피처 엔지니어링) ---")
print(classification_report(y_test, y_pred_xgb))
```
#### RESULT
|status|precision|recall|f1-score|support|
|:---:|:----:|:----:|:----:|:----:|
|0|       0.99 |     1.00  |    0.99 |     1932|
|1 |      0.86  |    0.81 |     0.83  |      68|

단순히 피처 엔지니어링만 수행해도 0.83의 매우 높은 f1 점수를 얻음을 알 수 있다. 이전의 RandomForest의 최고 성능이 0.77이었던 것에 비해 높은 점수이다. 정확도와 재현율 모두 고르게 0.8 이상으로 높은 점수를 얻었다.<br>

```python
### with Sampling
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# --- 1. 데이터 로드 및 전처리 ---
df = pd.read_csv('ai4i2020.csv')
X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
X = pd.get_dummies(X, columns=['Type'], drop_first=True)

# --- 2. 데이터 분할 ---
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)

# --- 3. SMOTE 오버샘플링 적용 ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# --- 4. XGBoost 모델 학습 및 평가 ---
# SMOTE를 사용했으므로 scale_pos_weight 파라미터는 제거합니다.
model_xgb = xgb.XGBClassifier(objective='binary:logistic',
                              use_label_encoder=False,
                              eval_metric='logloss',
                              random_state=42)

print("\n--- SMOTE 적용 데이터로 XGBoost 모델 학습 시작 ---")
model_xgb.fit(X_train_resampled, y_train_resampled)
print("--- 모델 학습 완료 ---")

print("\n--- 모델 평가 (테스트 데이터) ---")
y_pred = model_xgb.predict(X_test)
print("\n--- 최종 분류 리포트 (XGBoost + SMOTE) ---")
print(classification_report(y_test, y_pred))
```
#### RESULT
|status|precision|recall|f1-score|support|
|:---:|:----:|:----:|:----:|:----:|
|0|       0.99 |     0.98  |    0.99 |     1932|
|1 |      0.63  |    0.78 |     0.70  |      68|

이전에 RandomForest에서 사용했던 샘플링 기법을 사용하면 더 예측 정확도를 올릴 수 있을것이라 생각했으나 오히려 샘플링 때문에 예측에 혼돈이 생겨 결과가 낮아짐을 관찰가능하다.<br>
SMOTEENN을 사용해 노이즈를 제거해도 비슷한 결과를 얻었다.<br>
결국 이전의 RandomForest에서 논의한 것과 비슷하게 샘플링으로 생성된 데이터가 오히려 모델의 예측에 혼동을 주는 것으로 분석할 수 있다.<br>

GridSearchCV를 통해 파라미터들을 조정하며 최고 성능을 얻어보자.
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report

# --- 1. 데이터 로드 및 피처 엔지니어링 ---
df = pd.read_csv('ai4i2020.csv')
df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['Temp_Diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Overstrain'] = df['Tool wear [min]'] * df['Torque [Nm]']

# --- 2. 특성(X)과 타겟(y) 분리 ---
X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
X = pd.get_dummies(X, columns=['Type'], drop_first=True)

# --- 3. 데이터 분할 ---
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)

# --- 4. 하이퍼파라미터 탐색 범위 설정 ---
# XGBoost의 주요 하이퍼파라미터 후보들을 지정합니다.
param_grid = {
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# --- 5. 기본 XGBoost 모델 및 K-Fold 설정 ---
scale_pos_weight_value = np.sum(y_train == 0) / np.sum(y_train == 1)
xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                              scale_pos_weight=scale_pos_weight_value,
                              use_label_encoder=False,
                              eval_metric='logloss',
                              random_state=42)

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# --- 6. GridSearchCV 설정 및 실행 ---
# 고장(1) 클래스의 f1_score를 기준으로 최적의 모델을 찾도록 설정
grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=param_grid,
                           scoring='f1',
                           n_jobs=-1,
                           cv=kfold,
                           verbose=2)

print("--- [XGBoost 최종] GridSearchCV 하이퍼파라미터 탐색 시작 ---")
grid_search.fit(X_train, y_train)
print("--- 탐색 완료 ---")

# --- 7. 결과 확인 ---
print("\n--- [XGBoost 최종] 최적 하이퍼파라미터 ---")
print(grid_search.best_params_)

print("\n--- [XGBoost 최종] 교차 검증 최고 점수 (고장 클래스 F1) ---")
print(f"Best F1-Score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- 최종 분류 리포트 (XGBoost 최종) ---")
print(classification_report(y_test, y_pred))
```

#### RESULT
```
최적 하이퍼파라미터
{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}

```
|status|precision|recall|f1-score|support|
|:---:|:----:|:----:|:----:|:----:|
|0|       0.99 |     1.00  |    1.00 |     1932|
|1 |      0.89  |    0.82 |     0.85  |      68|

결과적으로 정확도와 재현율 모두 상승했다.<br>
하이퍼 파라미터 중 ```colsample_bytree```는 피처의 일부만 사용해 과적합을 방지하고, ```subsample```은 데이터 셋을 일부만 사용해 과적합을 방지한다.<br>
또한 GridSearchCV는 자체적으로 모델의 성능을 평가할 때 K겹 교차 검증을 통해 과적합이 일어나는 파라미터 셋을 선택하지 않게 해줘 과적합을 피할 수 있다.<br>
