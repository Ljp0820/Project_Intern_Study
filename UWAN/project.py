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