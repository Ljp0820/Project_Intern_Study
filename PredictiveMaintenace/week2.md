# 방학스터디 2주차

## Predictive Maintenance with RandomForest

### 1. RandomForest without Feature Engineering

랜덤 포레스트 기법은 신경망이 아닌 결정 트리를 사용한다. 결정 트리는 데이터를 분류하기 위한 질문을 가지는데, 어떤 질문이 데이터를 가장 잘 분류할 수 있는지가 모델의 성능을 결정한다. 그래서 랜덤 포레스트는 여러개의 결정 트리를 사용한다. 각각의 트리는 무작위하게 개성을 부여받아 각각 다른 질문을 생성한다. Bootstrap을 통해서 데이터셋을 무작위로 복원 추출해 나눠주거나, 특정 데이터만 트리에게 제공해서 각각 부여받은 데이터에 특화된 질문을 만들어낸다.<br>
랜덤 포레스트는 개별 데이터 특성의 임계값을 잘 분류하는 특징이 있다. 따라서 수치나 범주형 데이터에 강점이 있어, Predictive Maintenance 분야에 주로 사용되는 기법이라고 한다.<br>
CNN에서는 역전파를 통해 신경망의 가중치가 학습되지만, 랜덤 포레스트에서는 생성된 질문 후보에 데이터를 대입해서 질문의 순도를 측정한다. 순도는 질문이 데이터를 분류 했을 때, 그 결과가 실제 결과와 얼마나 잘 부합하는지를 나타내는 값이다. 순도가 높은 질문이 최적 질문으로 간주된다. 랜덤 포레스트에서는 이 과정을 반복해 최적의 질문을 찾아나가는 형식으로 학습이 진행된다.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('ai4i2020.csv')

X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']

X = pd.get_dummies(X, columns=['Type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)

model_rf = RandomForestClassifier(n_estimators=100,
                                  class_weight='balanced',
                                  random_state=42,
                                  n_jobs=-1)

print("--- 랜덤 포레스트 모델 학습 시작 ---")
model_rf.fit(X_train, y_train)
print("--- 모델 학습 완료 ---")

print("\n--- 모델 평가 (테스트 데이터) ---")
y_pred_rf = model_rf.predict(X_test)
print("\n--- 분류 리포트 (랜덤 포레스트) ---")
print(classification_report(y_test, y_pred_rf))
```
랜덤 포레스트에서는 데이터 전처리 시 정규화를 할 필요가 없다.<br>
```python
model_rf = RandomForestClassifier(n_estimators=100,
                                  class_weight='balanced',
                                  random_state=42,
                                  n_jobs=-1)
```
```n_estimators```는 트리의 개수를 지정하고, ```class_wight```를 모델을 구성하면서 바로 사용 가능하다. ```n_jobs```는 사용할 코어의 개수이다. -1인 경우 모든 코어를 사용한다.
```
--- 분류 리포트 (랜덤 포레스트) ---
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      1932
           1       0.94      0.47      0.63        68

    accuracy                           0.98      2000
   macro avg       0.96      0.73      0.81      2000
weighted avg       0.98      0.98      0.98      2000

```
결과를 보면 고장이 아닌 경우에 대해서는 매우 잘 예측한다. 그러나 고장인 경우, 정확도는 높으나 재현율이 낮다. 이는 모델이 고장이라고 판단한 것들은 실제로 고장이 맞는데, 고장인데 놓치는 경우가 많다는 것이다. 이런 경우 보통 클래스 가중치, 샘플링, 임계값 조절등의 방법으로 재현율을 높일 수 있다. 현재 모델에서는 클래스 가중치를 사용했지만 재현율이 낮음을 관찰 할 수 있다.<br>
재현율을 높이기 위해서 오버 샘플링 기법인 SMOTE를 사용했다. SMOTE는 고장인 경우를 바탕으로 임의로 고장인 데이터 셋을 생성해 고장과 고장이 아닌 경우의 데이터 비율을 맞춰준다. 임의로 생성한 데이터이기에 재현율은 높아지지만, 정확도가 낮아질 우려가 있다.<br>
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv('ai4i2020.csv')
X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
X = pd.get_dummies(X, columns=['Type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)

print("--- SMOTE 오버샘플링 적용 전 ---")
print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- SMOTE 오버샘플링 적용 후 ---")
print(pd.Series(y_train_resampled).value_counts())

model_rf_smote = RandomForestClassifier(n_estimators=100,
                                        random_state=42,
                                        n_jobs=-1)
                                        ## SOMTE를 사용하면 데이터 비율이 동일해지기 떄문에 클래스 가중치를 사용할 필요가 없다.

print("\n--- SMOTE 적용된 데이터로 랜덤 포레스트 모델 학습 시작 ---")
model_rf_smote.fit(X_train_resampled, y_train_resampled)
print("--- 모델 학습 완료 ---")

print("\n--- 모델 평가 (테스트 데이터) ---")
y_pred_rf_smote = model_rf_smote.predict(X_test)
print("\n--- 분류 리포트 (랜덤 포레스트 + SMOTE) ---")
print(classification_report(y_test, y_pred_rf_smote))
```
```
--- SMOTE 오버샘플링 적용 전 ---
0    7729
1     271
Name: count, dtype: int64

--- SMOTE 오버샘플링 적용 후 ---
0    7729
1    7729
Name: count, dtype: int64
```
샘플링 후 고장인 경우의 데이터가 생성된 것을 볼 수 있다.
```
--- 분류 리포트 (랜덤 포레스트 + SMOTE) ---
              precision    recall  f1-score   support

           0       0.99      0.98      0.98      1932
           1       0.52      0.68      0.59        68

    accuracy                           0.97      2000
   macro avg       0.75      0.83      0.78      2000
weighted avg       0.97      0.97      0.97      2000

```
결과를 보면 재현율이 0.47에서 0.68로 상승했지만, 우려한대로 정확도가 하락해 f1 점수는 낮아지는 결과를 보인다.<br>
오버 샘플링 기법의 정확도 하락의 문제를 해결하기 위해 하이브리드 샘플링 기법을 사용해보기로 했다. SMOTEENN은 SOMTE와 ENN을 결합한 하이브리드 샘플링 기법인데, SMOTE로 데이터를 생성하고 ENN을 노이즈를 제거해 SMOTE가 생성한 모호한 데이터를 제거한다.<br>
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN

df = pd.read_csv('ai4i2020.csv')
X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
X = pd.get_dummies(X, columns=['Type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)

print("--- SMOTEENN 하이브리드 샘플링 적용 전 ---")
print(pd.Series(y_train).value_counts())

smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

print("\n--- SMOTEENN 하이브리드 샘플링 적용 후 ---")
print(pd.Series(y_train_resampled).value_counts())

model_rf_se = RandomForestClassifier(n_estimators=100,
                                     random_state=42,
                                     n_jobs=-1)

print("\n--- SMOTEENN 적용된 데이터로 모델 학습 시작 ---")
model_rf_se.fit(X_train_resampled, y_train_resampled)
print("--- 모델 학습 완료 ---")

print("\n--- 모델 평가 (테스트 데이터) ---")
y_pred_rf_se = model_rf_se.predict(X_test)
print("\n--- 분류 리포트 (랜덤 포레스트 + SMOTEENN) ---")
print(classification_report(y_test, y_pred_rf_se))
```
```
--- SMOTEENN 하이브리드 샘플링 적용 전 ---
0    7729
1     271
Name: count, dtype: int64

--- SMOTEENN 하이브리드 샘플링 적용 후 ---
1    7297
0    6655
Name: count, dtype: int64
```
ENN으로 고장인 경우의 데이터 셋 일부가 제거되었음을 관찰할 수 있다.
```
--- 분류 리포트 (랜덤 포레스트 + SMOTEENN) ---
              precision    recall  f1-score   support

           0       0.99      0.95      0.97      1932
           1       0.35      0.76      0.48        68

    accuracy                           0.94      2000
   macro avg       0.67      0.86      0.73      2000
weighted avg       0.97      0.94      0.95      2000

```
정확도를 높이기 위해 SOMTEENN을 사용했지만, 오히려 정확도는 더 떨어지고 재현율이 높아졌다. ENN으로 삭제된 데이터의 개수가 1000개 이상인데, 이는 경계선 근처에서 실제로는 정상인데 모호하다고 판단해 삭제된 경우가 매우 많아 경계선 근처에서는 고장일 것이라고 모델이 학습하게 되기 때문이다.<br>
이를 해결하기 위해 파라미터 조정을 사용해보기로 했다. GridsearchCV를 통해 파라미터 셋을 생성해 모두 랜덤 포레스트로 예측을 진행하고 가장 나은 결과를 선택하는 방법을 사용했다.
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```
결과적으로는 기본으로 사용했던 파라미터 셋이 가장 최적의 결과였다.<br>
정확도를 높이기 위해 임계값을 조정해보기로 했다. 기본으로는 고장일 확률이 0.5 이상인 경우 고장으로 예측하기로 했는데, 이를 높혀 정확도를 높이고자 했다. 임계값을 0.5부터 0.9까지 0.1씩 키우며 결과를 관측해보기로 했다.<br>

![Result]()

결과를 보면, 일단은 SMOTE가 SMOTEENN보다 더 좋은 성능을 일관적으로 보인다. 임계값은 0.5에서 0.7 사이가 가장 좋은 결과를 보인다.<br>
그러나 고장인 경우의 f1 점수가 0.6을 넘지 못하는 여전히 아쉬운 성능을 보인다. 이를 해결 하기 위해 feature engineering을 사용해보기로 했다.

### 2. RandomForest With Feature Engineering

데이터셋 Kaggle 사이트를 참고하면 각 데이터들의 설명이 나오는데, 각 고장에 대한 설명이 나온다.
```
tool wear failure (TWF): the tool will be replaced of fail at a randomly selected tool wear time between 200 - 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
heat dissipation failure (HDF): heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tools rotational speed is below 1380 rpm. This is the case for 115 data points.
power failure (PWF): the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.
overstrain failure (OSF): if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.
random failures (RNF): each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.
If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. It is therefore not transparent to the machine learning method, which of the failure modes has caused the process to fail.
```
각각의 고장 유형이 어떤 식으로 발생하는 지 알려주는데, 이 정보를 바탕으로 기존의 데이터셋에서 추가적인 열을 생성하는 것을 feature engineering이라고 한다.<br>

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('ai4i2020.csv')

print("--- 피처 엔지니어링 수행 ---")
# Power 특성 (Torque * Rotational Speed)
df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
# Temperature Difference 특성
df['Temp_Diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
# Overstrain 특성 (Tool wear * Torque)
df['Overstrain'] = df['Tool wear [min]'] * df['Torque [Nm]']
print("Power, Temp_Diff, Overstrain 특성 3개 추가 완료")

features_to_analyze = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]',
    'Power [W]',             
    'Temp_Diff [K]',         
    'Overstrain',            
    'Machine failure'
]
df_analyzed = df[features_to_analyze]

plt.figure(figsize=(12, 10))
sns.heatmap(df_analyzed.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap After Feature Engineering')
plt.tight_layout()
plt.savefig('heatmap_after_feature_engineering.png')
```
![FeatureEngineering]()

데이터 셋 설명을 바탕으로 ```Power```, ```Temp_Diff```, ```Overstrain``` 항목을 새로 생성해 상관계수 분석을 진행했다.<br>
선형적인 상관 관계는 딱히 보이지 않는다. 대부분이 0.1-0.2 정도 사이에 머물러 있음을 알 수 있다.<br>
feature engineering을 수행한 데이터 셋을 바탕으로 임계값을 조정해가면서 결과를 관찰해보자.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# --- 1. 데이터 로드 ---
df = pd.read_csv('ai4i2020.csv')

# --- 2. 피처 엔지니어링 ---
print("--- 피처 엔지니어링 수행 ---")
df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['Temp_Diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Overstrain'] = df['Tool wear [min]'] * df['Torque [Nm]']
print("Power, Temp_Diff, Overstrain 특성 3개 추가 완료")

# --- 3. 특성(X)과 타겟(y) 분리 ---
X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
X = pd.get_dummies(X, columns=['Type'], drop_first=True)

# --- 4. 데이터 분할 ---
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)

# --- 5. 실험 설정 ---
sampling_methods = {
    'SMOTE': SMOTE(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42)
}
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
results = []

# --- 6. 모델 학습 및 임계값별 성능 평가 ---
for name, method in sampling_methods.items():
    print(f"\n--- [{name} + Feature Engineering] 적용 및 모델 학습 시작 ---")
    
    X_train_resampled, y_train_resampled = method.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    for th in thresholds:
        y_pred = (y_pred_proba >= th).astype(int)
        
        p_class0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        r_class0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_class0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        
        p_class1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        r_class1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_class1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        results.append({
            'Method': f"{name}+FE", 'Threshold': th,
            'Precision_0': p_class0, 'Recall_0': r_class0, 'F1_0': f1_class0,
            'Precision_1': p_class1, 'Recall_1': r_class1, 'F1_1': f1_class1
        })

# --- 7. 결과 표(Table)로 출력 ---
results_df = pd.DataFrame(results)
print("\n--- [최종] 임계값별 성능 비교표 ---")
print(results_df)

# --- 8. 결과 그래프로 시각화 ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Performance Metrics with Feature Engineering', fontsize=16)

# SMOTE + FE 그래프
smote_df = results_df[results_df['Method'] == 'SMOTE+FE']
axes[0, 0].plot(smote_df['Threshold'], smote_df['Precision_1'], marker='o', label='Precision (Failure)')
axes[0, 0].plot(smote_df['Threshold'], smote_df['Recall_1'], marker='o', label='Recall (Failure)')
axes[0, 0].plot(smote_df['Threshold'], smote_df['F1_1'], marker='o', label='F1-Score (Failure)')
axes[0, 0].set_title('SMOTE + Feature Engineering - Failure Class (1)')
axes[0, 0].grid(True)
axes[0, 0].legend()

axes[0, 1].plot(smote_df['Threshold'], smote_df['Precision_0'], marker='s', label='Precision (Normal)')
axes[0, 1].plot(smote_df['Threshold'], smote_df['Recall_0'], marker='s', label='Recall (Normal)')
axes[0, 1].plot(smote_df['Threshold'], smote_df['F1_0'], marker='s', label='F1-Score (Normal)')
axes[0, 1].set_title('SMOTE + Feature Engineering - Normal Class (0)')
axes[0, 1].grid(True)
axes[0, 1].legend()

# SMOTEENN + FE 그래프
smoteenn_df = results_df[results_df['Method'] == 'SMOTEENN+FE']
axes[1, 0].plot(smoteenn_df['Threshold'], smoteenn_df['Precision_1'], marker='o', label='Precision (Failure)')
axes[1, 0].plot(smoteenn_df['Threshold'], smoteenn_df['Recall_1'], marker='o', label='Recall (Failure)')
axes[1, 0].plot(smoteenn_df['Threshold'], smoteenn_df['F1_1'], marker='o', label='F1-Score (Failure)')
axes[1, 0].set_title('SMOTEENN + Feature Engineering - Failure Class (1)')
axes[1, 0].grid(True)
axes[1, 0].legend()

axes[1, 1].plot(smoteenn_df['Threshold'], smoteenn_df['Precision_0'], marker='s', label='Precision (Normal)')
axes[1, 1].plot(smoteenn_df['Threshold'], smoteenn_df['Recall_0'], marker='s', label='Recall (Normal)')
axes[1, 1].plot(smoteenn_df['Threshold'], smoteenn_df['F1_0'], marker='s', label='F1-Score (Normal)')
axes[1, 1].set_title('SMOTEENN + Feature Engineering - Normal Class (0)')
axes[1, 1].grid(True)
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('final_threshold_comparison.png')
print("\n> 최종 성능 비교 그래프가 'final_threshold_comparison.png'로 저장되었습니다.")
```
![FeatureEngineeringResult]()

결과를 보면, 여전히 SMOTE를 사용한 경우가 SMOTEENN을 사용한 경우에 비해 좋은 결과를 보인다. SMOTE에서 임계값 0.5, 0.6 정도에서 가장 좋은 성능을 보임을 알 수 있다.
```
--- 최종 분류 리포트 (Random Forest 최종) ---
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1932
           1       0.73      0.81      0.77        68

    accuracy                           0.98      2000
   macro avg       0.86      0.90      0.88      2000
weighted avg       0.98      0.98      0.98      2000
```
0.6에서 고장인 경우 정확도 0.73, 재현율 0.81으로 이전에 비해 아주 우수한 성능을 보인다. 그러나 아직 모델이 완벽하게 예측한다고 말하기에는 다소 부족한 수치이다.<br>
다음주에는 더 좋은 방법이라고 하는 XGboost를 사용할 경우 성능이 더 좋은지 확인해보기로 했다.