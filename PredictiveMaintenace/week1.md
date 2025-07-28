# 방학스터디 1주차

## Predictive Maintenance with CNN

### 1. Correlation Analysis<br>
먼저 데이터 셋에서 예측하려는 데이터와, 예측에 필요한 데이터들의 상관관계를 분석해보자.<br>
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ai4i2020.csv')

numeric_features = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]',
    'Machine failure'
]
df_numeric = df[numeric_features]

plt.figure(figsize=(10, 8))

sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Predictive Maintenance Dataset Correlation Heatmap')
plt.tight_layout()
plt.show()
```
![Result](https://github.com/Ljp0820/Project_Intern_Study/blob/main/PredictiveMaintenace/ai4i2020_correlation_heatmap.png)<br>
결과를 보면, Machine failure와 상관 관계가 높은 데이터는 딱히 보이지 않음을 알 수 있다.<br>
높은 상관 계수를 보이는 데이터가 없다는 것이 모델이 예측을 잘할 수 없다를 의미하지는 않는다. 상관계수는 변수들 간의 선형 관계만 고려하는 지표이기에 U 자형과 같은 비선형 관계를 보여줄 수 없다.<br>

### 2. CNN(Basic)
일단 선형적인 관계가 드러나지 않음을 확인했으니, 기본적인 CNN으로 모델이 얼마나 잘 예측하는지 살펴보자<br>
데이터의 정규화를 표준화로 수행하고, 1개 층이 존재하는 기본적인 CNN 모델이다.<br>
성능 지표로는 Precision(정확도), Recall(재현율), f1-score를 사용했다.<br>
정확도는 모델이 고장/고장이 아니라고 예측한 것 중 실제로 그러한 비율이다.<br>
재현율은 실제로 고장/고장이 아닌 것 중 모델이 그렇게 예측한 비율이다.<br>
f1-score는 정확도와 재현율을 종합한 지표이다. 정확도와 재현율은 보통 반비례 관계를 가지기에 두 지표의 조화평균을 사용해 얼마나 모델이 균형있게 예측하는지 알 수 있다.<br>
```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

df = pd.read_csv('ai4i2020.csv')

X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
X = pd.get_dummies(X, columns=['Type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- CNN 모델 학습 시작 ---")

history = model.fit(X_train_cnn, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)
print("--- 모델 학습 완료 ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.savefig('training_history.png')
print("\n> 학습 과정 그래프가 'training_history.png'로 저장되었습니다.")

print("\n--- CNN 모델 최종 평가 (테스트 데이터) ---")
y_pred_proba = model.predict(X_test_cnn)
y_pred = (y_pred_proba > 0.5).astype("int32")
print("\n--- 최종 분류 리포트 (CNN) ---")
print(classification_report(y_test, y_pred))
```

```
--- 최종 분류 리포트 (CNN) ---
              precision    recall  f1-score   support

           0       0.98      0.99      0.99      1932
           1       0.67      0.49      0.56        68

    accuracy                           0.97      2000
   macro avg       0.83      0.74      0.78      2000
weighted avg       0.97      0.97      0.97      2000
```
결과를 보면 고장이 아닌 것에 대한 정확도와 재현율은 매우 높은 것을 볼 수 있다. 그러나 고장에 대한 정확도와 재현율은 매우 낮은데, 이는 모델이 거의 대부분을 고장이 아니라고 판단해버린 것으로 분석 가능하다. 즉 CNN이 데이터를 통해 고장/비고장을 예측할 수 없었고, 비고장인 비율이 단순히 높아서 그냥 고장이 아니라고 예측하는 것이 높은 정확도를 받을 수 있다고 학습한 결과이다.<br>
즉, 주어진 원본 데이터 셋과 기본적인 CNN 모델로는 예측이 잘 되지 않음을 알 수 있다.<br>

### 3. CNN(Advanced)
CNN 모델의 성능 향상을 위해 여러가지 기법을 추가해서 학습을 진행했다.<br>
```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df = pd.read_csv('ai4i2020.csv')

X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']
X = pd.get_dummies(X, columns=['Type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i : weights[i] for i in range(len(weights))}

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

print("\n--- 5-Conv CNN 모델 학습 시작 (피처 엔지니어링 제외) ---")
history = model.fit(X_train_cnn, y_train,
                    epochs=100, 
                    batch_size=32,
                    validation_split=0.2,
                    class_weight=class_weights,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)
print("--- 모델 학습 완료 ---")

print("\n--- CNN 모델 최종 평가 (테스트 데이터) ---")
y_pred_proba = model.predict(X_test_cnn)
y_pred = (y_pred_proba > 0.5).astype("int32")
print("\n--- 최종 분류 리포트 (5-Conv CNN, 피처 엔지니어링 제외) ---")
print(classification_report(y_test, y_pred))
```
예측 성능의 향상을 위해선 다양한 기법을 사용할 수 있는데, 크게 모델 구조, 과적합 방지, 학습 최적화로 나눌 수 있다.<br>
먼저 모델 구조부터 살펴보자.<br>
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train_cnn.shape[1], 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
```tf.keras.layers.BatchNormalization()```를 통해 배치 정규화를 사용했다. 배치 정규화는 Layer를 통과한 값들을 평균이 0이고, 표준편차가 1로 정규화를 시켜주는 기법이다. 배치 정규화를 통해 모델은 학습을 빠르게 할 수 있고, 데이터가 레이어를 통과하면서 편향된 분포를 가지거나 불안정해지는 것을 방지해 안정적인 학습을 할 수 있다.<br>
또한, 이전의 기본 CNN 모델에 비해 5개의 레이어를 추가해 모델이 특징을 더 잘 추출할 수 있게 했다.<br><br>

과적합 방지를 위해 Dropout과 Callback의 조기종료를 사용했다. 과적합은 모델이 학습 데이터 셋에만 맞춰지게 학습이 되는 것을 말한다.<br>
```tf.keras.layers.Dropout(0.5)```을 통해 50%의 노드의 출력을 0으로 만들어 과적합을 방지한다.<br>
Callback의 조기종료 기능은 연속해서 Val_loss가 개선되지 않는 경우 모델이 과적합 상태에 돌입했다고 판단하고 학습을 조기 종료 시킨 후 Val_loss가 가장 좋았던 모델로 돌려준다.<br>
```early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)```을 사용하여 구현했다.<br><br>

학습 최적화를 위해 클래스 가중치와, 학습률 스케쥴러를 사용한다.<br>
클래스 가중치는 재현율을 높힐 수 있는 기법이다. 특히 현재 데이터 셋과 같이 특정 상황에 대한 데이터가 부족한 경우에 효과적으로 작동가능하다. 현재 데이터 셋에서 재현율이 낮은 이유는 대부분의 경우가 고장이 아니기에 고장이 아니라고 예측하면 최소 95% 이상은 정확하게 예측이 가능해서 모델이 거의 모든 경우에 대해서 고장이 아니라고 예측하기에 발생한다.<br>
클래스 가중치는 이때 고장인 경우에 대해 가중치를 부여해 큰 벌점을 받게 한다. 결과적으로 모델은 고장 상황에서의 패턴에 집중해 고장에 대한 재현율을 향상 시킬 수 있다.
```python
# 클래스 가중치 코드
# class_weight.compile_class_weight에서 각각의 가중치를 계산. 리스트 형태로 반환
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i : weights[i] for i in range(len(weights))}
```
클래스 가중치 기법은 위와 같이 사용하는데, ```class_weight.compute_class_weight```는 y에 대해 가중치를 계산해 리스트 형태로 반환해 모델은 이 가중치를 바탕으로 점수를 매겨 고장 상황에 대해 집중하도록 한다.<br>
학습률 스케쥴러는 동적 학습률을 설정한 것과 유사하게 모델에 영향을 준다. ```reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)```에서 patience 만큼 val_loss 성능의 개선이 없을 경우 factor를 learning rate에 곱해 학습률을 낮춘다. 이는 학습 초기에는 학습률을 크게 설정해 빠르게 수렴하게 하고, 학습이 진행될 수록 학습률은 낮춰 세밀하게 학습을 조정할 수 있게 해준다.
### 4. Results
```
--- 최종 분류 리포트 (5-Conv CNN, 피처 엔지니어링 제외) ---
              precision    recall  f1-score   support

           0       0.98      0.97      0.98      1932
           1       0.37      0.47      0.41        68

    accuracy                           0.95      2000
   macro avg       0.67      0.72      0.69      2000
weighted avg       0.96      0.95      0.96      2000
```
흥미롭게도 결과를 분석해보면, 많은 기법들을 넣은 CNN 모델이 오히려 기본적인 CNN에 비해서 성능이 낮게 나오는 것을 알 수 있다. 특히 고장(1)에 대한 성능을 주목해야하는데, 기본 CNN에서는 고장에 대한 정밀도가 0.67인 반면, 5-Conv CNN에서는 정밀도가 0.37이다. 그리고 재현율도 0.49에서 0.47로 하락했음을 관찰할 수 있다. 왜 이러한 결과가 나타났을까?<br>
아무래도 가장 큰 원인은 5개의 layer를 사용한 것일 수 있다. 복잡하지 않은 데이터 셋에 5개나 되는 레이어를 사용해 노이즈나 우연에 의한 잘못된 패턴을 학습하고, 클래스 가중치를 통해 고장을 더 적극적으로 예측하게 해서 정밀도와 재현율 모두가 하락했다고 분석할 수 있다.<br>
즉 데이터 셋의 복잡도와 비슷한 수준의 복잡도를 지닌 CNN을 사용하는 것이 좋다는 결론을 얻을 수 있다. 그렇다면 Layer를 2개로 줄인 CNN의 예측 결과는 어떨까?
```
--- 최종 분류 리포트 (2-Conv CNN, 피처 엔지니어링 제외) ---
              precision    recall  f1-score   support

           0       0.99      0.95      0.97      1932
           1       0.37      0.85      0.52        68

    accuracy                           0.95      2000
   macro avg       0.68      0.90      0.74      2000
weighted avg       0.97      0.95      0.96      2000
```
실제로 레이어 수를 줄이니 정밀도는 그대로지만 재현율 수치가 많이 높아졌음을 알 수 있다. 얕은 모델의 깊이가 단순한 데이터를 잘 학습했고, 클래스 가중치로 인해 적극적으로 고장을 예측한 결과가 나타났음을 알 수 있다. 그러나 재현율을 향상시켰지만, 정밀도는 낮은 수치를 유지해 결국 f1-score는 0.5정도의 낮은 값을 유지한다. 이는 결국 CNN이 주어진 데이터 셋을 제대로 학습할 수 없음을 보이는 결과이다. CNN이 아닌 다른 모델을 사용해야함을 알 수 있다.
