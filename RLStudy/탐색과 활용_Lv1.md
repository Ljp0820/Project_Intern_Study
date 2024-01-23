## 탐색과 활용 연습
~~뭐할까 고민 많이 해봤는데 제일 재미있는건 역시 코드 짜는게 아닐까요?(아님말고)~~<br>
주피터 노트북에서 권장(.ipynb)<br>
lv 1은 기본적인 구조 제시가 되어있는데, 아 이건 너무 쉽다는 생각이 드실 수도 있어서 lv 2를 준비했습니다.<br>
자신 있다면 lv 2로 시작하시길 추천
```python
# 필요한 라이브러리와 조건 초기화
import numpy as np
import matplotlib.pyplot as plt

time_step = 10000
slot_prob = [0.1, 0.15, 0.1, 0.65]
```
### Epsilon-Greedy

```python

# 일반 입실론 방법 데이터 컨테이너
selected_eps = [0 for i in range(len(slot_prob))]
reward = [0 for i in range(len(slot_prob))]
avg_reward = [0 for i in range(len(slot_prob))]
total_reward = []

# 입실론 값
eps = 0.7

for i in range(time_step):

    # 1. 액션 선택
    # 탐색의 경우 랜덤한 선택
    # 활용의 경우 보상이 가장 큰 행동 선택
    # 그냥 누적 보상이 가장 큰 행동을 선택하는 경우에 초기에 운이 좋게 얻은 보상을 과대평가하게 됩니다.
    # 그러므로 평균 보상이 가장 큰 행동을 선택함으로 보상 편향을 극복 가능합니다.
    # Hint. 리스트의 요소 중 가장 큰 요소의 인덱스를 추출하는 방법에 대해 생각해보기.


    
    selected_eps[act] += 1

    # 2. 보상 부여
    # 선택한 slot이 당첨되면 +1의 보상을, 그렇지 않다면 0의 보상 부여



    # 3. 평균 보상과 총 누적 보상율 업데이트
    avg_reward[act] = reward[act]/selected_eps[act] if selected_eps[act]>0 else 0
    
    total_reward.append(sum(reward)/(i+1))

# 시각화
plt.plot([i+1 for i in range(time_step)], total_reward)
plt.show()

```

### Thomson Sampling

```python
# 톰슨 샘플링 컨테이너
num_of_sel=[0 for i in range(len(slot_prob))]
reward_ts = [[0, 0] for i in range(len(slot_prob))]
avg_reward_ts = [0 for i in range(len(slot_prob))]
total_reward_ts=[]

for i in range(time_step):

    # 1. 액션 선택
    # 각 슬롯마다 베타 분포를 따르는 난수 추출
    # 가장 큰 값을 가지는 슬롯이 액션
    
    

    # 슬롯 선택 횟수 업데이트
    num_of_sel[selected_arm]+=1

    # 2. 보상 부여 및 베타 함수 파라미터 업데이트
    # 선택한 슬롯의 확률을 바탕으로 슬롯의 성공 여부 확인
    # 성공하는 경우 보상 +1
    # 성공하는 경우와 실패하는 경우 각각 보상 리스트에 횟수 추가로 베타 함수 파라미터 업데이트




    avg_reward_ts[selected_arm] = reward_ts[selected_arm][0]/num_of_sel[selected_arm] if num_of_sel[selected_arm] > 0 else 0

    total_reward_ts.append(sum(reward_ts[j][0] for j in range(len(slot_prob)))/(i+1))

# 전체 선택횟수 확인과 시각화
print(num_of_sel)
plt.plot(np.arange(1, time_step+1), total_reward_ts)
plt.show()
```

### UCB
혹시 두 개로 부족할까봐...
```python
# UCB 컨테이너
num_of_sel_ucb = [0 for i in range(len(slot_prob))]
reward_ucb = [0 for i in range(len(slot_prob))]
avg_reward_ucb = [0 for i in range(len(slot_prob))]
total_reward_ucb = []

score_ucb = [0 for i in range(len(slot_prob))]

for i in range(time_step):

    # UCB 점수 업데이트
    # np.sqrt, np.log사용
    

    
    # UCB 점수를 바탕으로 행동 선택하기
    

    # 보상 부여
    
    

    avg_reward_ucb[action] = reward_ucb[action]/num_of_sel_ucb[action] if num_of_sel_ucb[action] > 0 else 0

    total_reward_ucb.append(sum(reward_ucb)/(i+1))

# 시각화
plt.plot(np.arange(1, time_step+1), total_reward_ucb)
plt.show()
```