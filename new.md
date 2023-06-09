 UWAN
 =====

과제1
-----

## Poisson Distribution과 Exponential Distribution
교수님께서 첫 번째로 주신 과제는 포아송 분포와 지수 분포의 관계에 대한 문제였다. <br>
1. 지수분포는 연속확률 분포로, $f=e^{-\lambda x}$와 같은 확률 밀도 함수를 가진다. 이때 $\lambda$는 단위 시간 동안 발생하는 사건의 횟수다. 지수분포의 기댓값은 $E(x)=\frac{1}{\lambda}$ 인데, 이를 통해 지수 분포는 어떤 사건의 대기시간에 대한 분포라는 점을 알 수 있다.<br>
2. 포아송 분포는 $f=e^어떤 사건이 일정 시간동안 일어날 횟수에 대한 분포이다.<br>
그러므로 일정시간동안 특정 lambda값을 지수분포를 따르는 어떤 사건의

```python
    import numpy as np
import matplotlib.pyplot as plt

lamb=5 # 단위시간 당 발생 횟수
time=10 # 관찰할 시간
list_deltaT=np.array([]) # time interval을 저장하는 넘파이 어레이, time interval 관찰할 일 있을까봐 추가해놓긴 했는데,,,
num_of_occur=np.array([]) # 발생 횟수 저장하는 넘파이 어레이
poisson_list=np.array([]) # 포아송 분포에 따른 발생 횟수 저장하는 어레이
iter=1000 # 반복횟수

for step in range(iter):
    count=0 # 발생횟수
    init=0 # 초기 시간 0초
    while True:
        deltaT=np.random.exponential(1/lamb) # 지수분포를 따르는 랜덤한 time interval 생성
        init+=deltaT # 시간에 계속 더해나가는 코드
        list_deltaT=np.append(list_deltaT, deltaT)
        count+=1
        if init>10: # time 10을 넘어서 발생 한 경우
            count-=1 # 발생횟수 -1
            list_deltaT=np.delete(list_deltaT, count) # 마지막으로 추가된 time interval 삭제
            num_of_occur=np.append(num_of_occur, count) # 발생 횟수를 append
            poisson=np.random.poisson(lam=lamb*time) # 포아송 분포를 따르는 발생 횟수 랜덤하게 생성
            poisson_list=np.append(poisson_list, poisson) # 위의 값 저장
            plt.subplot(2,2,1)
            plt.plot(step, np.sum(num_of_occur)/(step+1),'r.') # step마다 평균 발생횟수 점으로 그리기
            plt.subplot(2,2,3)
            plt.plot(step, count, "b.") # step마다의 지수함수를 따르는 발생횟수
            plt.subplot(2,2,4)
            plt.plot(step, poisson, "g.") # step 마다 포아송 분포를 따르는 발생횟수
            break

total_occur=np.sum(num_of_occur) # 총 발생횟수
avg=total_occur/iter # 발생 횟수의 평균


exp_avg=round(avg,4) # 지수분포의 평균, 분산, 표준편차
exp_std=round(np.std(num_of_occur),4)
exp_var=round(np.var(num_of_occur),4)

pois_avg=round(np.sum(poisson_list)/iter,4) #포아송 분포의 평균, 분산, 표준편차
pois_std=round(np.std(poisson_list),4)
pois_var=round(np.var(poisson_list),4)

plt.subplot(2,2,1)
plt.axhline(time*lamb, 0, 1, color='green')
plt.axhline(avg, color='blue', linestyle='--')
plt.subplot(2,2,2)
plt.boxplot([num_of_occur, poisson_list]) # 사분위수 관찰하려고 추가한겁니다.
plt.xticks([1,2],["Exp", "Poisson"])
plt.subplot(2,2,3)
plt.xlabel('avg='+str(exp_avg)+'    var='+str(exp_var)+'    std='+str(exp_std))
plt.subplot(2,2,4)
plt.xlabel('avg='+str(pois_avg)+'    var='+str(pois_var)+'    std='+str(pois_std))
plt.show()
```

과제 2
-----
```python
import numpy as np
import matplotlib.pyplot as plt

# non slotted ver

class UWAN:
    def __init__(self):       

        self.lam=np.random.randint(3,8)                     #단위시간 당 발생 횟수
        self.receive_period=round(np.random.rand(), 3)      #수신 주기
        self.awake_time=self.receive_period*0.4             #awake time 수신 주기의 40퍼센트로 설정
        self.success=0                                      #성공횟수
        self.counter=0                                      #block counter, receive() 에서 사용
        self.ACK=False                                      #성공여부
        self.trans_time=0.05                                #전송시간(수중)
        self.sink_time=0
        self.sensor_time=0
        self.delta=0                                        #실패했을 경우 가장 가까운 awake time과의 시간차
        self.direct=0                                       #실패했을 경우 가장 가까운 awake time 왼쪽에 있는지 오른쪽에 있는지 구분 0=left, 1=right
        
        if self.receive_period <= 0.01:                     #주기 반올림해서 0이 되는 경우 조정
            self.receive_period=0.01
            self.awake_time=self.awake_time*0.4
        

    def step(self):                                         #step() return은 성공여부, 실패시 False, 시간차, 방향
        UWAN.send_msg(self)
        UWAN.receive_msg(self)

        return self.ACK if self.ACK==True else self.ACK, round(self.delta, 5), self.direct
            
    def send_msg(self):                                     #메세지 생성:exp함수에 따른 발생
        
        occur_time=round(np.random.exponential(1/self.lam),4)
        self.sink_time+=occur_time
        self.sensor_time+=occur_time+self.trans_time
        ax.plot(self.sensor_time, 5, 'bo')

    def receive_msg(self):                                  #메세지 수신, 수신된 메세지가 awake_time에 들어와서 제대로 수신되었는지 판단

        left_delta=1000
        right_delta=1000
        block=self.awake_time+self.receive_period

        while True:
            ax.plot(block*self.counter, 5, 'go')
            ax.plot(block*self.counter+self.awake_time, 5 , 'ro')
            if self.counter*block <= self.sensor_time and self.sensor_time <=self.counter*block+self.awake_time:
                self.delta=0
                self.ACK=True
                self.success+=1
                break
            elif self.sensor_time<self.counter*block and self.counter*block-self.sensor_time<block:
                left_delta=abs((self.counter-1)*block+self.awake_time-self.sensor_time)
                right_delta=abs(self.counter*block-self.sensor_time)
                self.delta=min(left_delta, right_delta)
                if self.delta==left_delta:
                    self.direct=0
                else:
                    self.direct=1
                self.ACK=False
                break
            else:
                self.counter+=1
                self.ACK=False
        
env=UWAN()

print(env.lam, end=' ')
print(env.receive_period, end=' ')
print(env.awake_time)

fig=plt.figure(figsize=(20,9))
ax=fig.add_subplot(1,1,1)

for j in range(10):

    print(env.step())

print(env.success)
plt.show()
```

```python
    import numpy as np
import matplotlib.pyplot as plt

# non slotted ver

class UWAN:
    def __init__(self):       

        self.lam=np.random.randint(3,8)                     #단위시간 당 발생 횟수
        self.receive_period=round(np.random.rand(), 3)      #수신 주기
        self.awake_time=round(self.receive_period*0.4, 3)   #awake time 수신 주기의 40퍼센트로 설정
        self.success=0                                      #성공횟수
        self.counter=0                                      #block counter, receive() 에서 사용
        self.ACK=False                                      #성공여부
        self.trans_time=0.05                                #전송시간(수중)
        self.sink_time=0
        self.sensor_time=0
        self.delta=0                                        #실패했을 경우 가장 가까운 awake time과의 시간차
        self.direct=0                                       #실패했을 경우 가장 가까운 awake time 왼쪽에 있는지 오른쪽에 있는지 구분 0=left, 1=right
        
        if self.receive_period <= 0.01:                     #주기 반올림해서 0이 되는 경우 조정
            self.receive_period=0.01
            self.awake_time=self.awake_time*0.4
        

    def step(self):                                         #step() return은 성공여부, 실패시 False, 시간차, 방향
        UWAN.send_msg(self)
        UWAN.receive_msg(self)

        return self.ACK if self.ACK==True else self.ACK, round(self.delta, 5), self.direct
            
    def send_msg(self):                                     #메세지 생성:exp함수에 따른 발생
        
        occur_time=round(np.random.exponential(1/self.lam),3)
        if occur_time == 0:
            occur_time=0.001
            
        self.sink_time+=occur_time
        self.sensor_time+=occur_time+self.trans_time

    def receive_msg(self):                                  #메세지 수신, 수신된 메세지가 awake_time에 들어와서 제대로 수신되었는지 판단

        left_delta=1000
        right_delta=1000
        block=self.awake_time+self.receive_period

        while True:
            if self.counter*block <= self.sensor_time and self.sensor_time <=self.counter*block+self.awake_time:
                self.delta=0
                self.ACK=True
                self.success+=1
                break
            elif self.sensor_time<self.counter*block and self.counter*block-self.sensor_time<block:
                left_delta=abs((self.counter-1)*block+self.awake_time-self.sensor_time)
                right_delta=abs(self.counter*block-self.sensor_time)
                self.delta=min(left_delta, right_delta)
                if self.delta==left_delta:
                    self.direct=0
                else:
                    self.direct=1
                self.ACK=False
                break
            else:
                self.counter+=1
                self.ACK=False
        
env=UWAN()

print(1/env.lam, end=' ')
print(env.receive_period, end=' ')
print(env.awake_time)

for j in range(1000):

    env.step()

print(env.success/1000)
```