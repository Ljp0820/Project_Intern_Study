import numpy as np
import matplotlib.pyplot as plt

# non slotted ver

class UWAN:
    def __init__(self):       

        self.lam=np.random.randint(3,8)
        self.receive_period=round(np.random.rand(), 3)
        self.awake_time=round(0.1*np.random.rand(),3)        
        self.success=0
        self.counter=0
        self.ACK=False
        self.trans_time=0.05
        self.sink_time=0
        self.sensor_time=0
        self.delta=0
        self.direct=0

        if self.receive_period <= 0.01:
            self.receive_period=0.01
        if self.awake_time <= 0.01:
            self.awake_time = 0.01
        

    def reset(self):    #epi 끝나면 변수들이 update되는데, reset해주는 함수
        self.counter=0
        pass

    def step(self):
        UWAN.send_msg(self)
        UWAN.receive_msg(self)

        return self.ACK, self.delta, self.direct
            
    def send_msg(self):
        
        occur_time=round(np.random.exponential(1/self.lam),4)
        self.sink_time+=occur_time
        self.sensor_time+=occur_time+self.trans_time
        ax.plot(self.sensor_time, 5, 'bo')

    def receive_msg(self):

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

fig=plt.figure()
ax=fig.add_subplot(1,1,1)

for j in range(10):

    print(env.step())

print(env.success)
plt.show()