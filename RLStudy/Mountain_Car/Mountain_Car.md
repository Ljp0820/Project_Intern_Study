# Mountain Car
## 서론(변명)
1. 먼저 학습이 되는가에 대해 생각을 해 보았다. 앞선 문제들은 현재의 상태에서 바로 Reward를 부여하는것에 대해 문제가 없었는데, 이번 문제는 음의 방향으로 반동을 주었다가 양의 방향으로 올라가야 하는데, 이러한 움직임을 어떻게 학습시킬 지에 대해 고민해보았다. 즉, 어떻게 Reward를 부여할 것인가가 문제였다. 처음에는 위치에너지를 생각해서 ```Reward=(n_state[0]+0.5)**2 + n_state[1]```와 같이 높이와 속도에 따른 보상을 부여해보기로 했다. 그 후 다른 Reward 함수들을 적용해서 많이 시행 해 보았는데, 결과적으로 학습이 잘 되는거 같진 않았다.
2. 두 번쨰로 많이 했던 고민은, 자원에 대한 문제라고 할 수 있을 것이다. 처음에는 순전파와 역전파를 Tensorflow에 있는 ```model.predict(), model.fit()```을 사용했는데, ```epi``` 40정도에서 커널이 죽어버리는 문제가 항상 발생했다. 이후 클래스를 따로 만들어서 사용하니 이러한 문제는 해결 되었다.

## 본론

### 기본 코드 구조

```python
import gymnasium as gym
import random
import numpy as np
import tensorflow as tf
from keras import datasets, layers, Model

env = gym.make('MountainCar-v0')

class DQN(Model):

    def __init__(self):

        super(DQN, self).__init__()
        self.d1 = layers.Dense(32, input_dim=2, activation='relu')
        self.d2 = layers.Dense(16, activation='relu')
        self.d3 = layers.Dense(16, activation='relu')
        self.d4 = layers.Dense(3, activation='linear')
        self.optimizer = tf.keras.optimizers.Adam(0.001)

    def call(self, x):

        x=self.d1(x)
        x=self.d2(x)
        x=self.d3(x)
        x=self.d4(x)

        return x
    
model=DQN()

time_spend=[]
memory=[]
epi=1000
time=200
eps=1.0
batch=64
count=0

for i in range(epi):
    
    state, _ = env.reset()
    eps=eps*0.95
    
    if len(memory)>1000:
        memory.pop(0)
    
    for t in range(time):
        if eps <= np.random.rand():
            action=np.argmax(model.call(np.array([state])))
        else:
            action=np.random.randint(0,3)

        n_state, reward, done, _, _ = env.step(action)

        if done:
            reward=10
        elif n_state[0]>=state[0] and action==2:
            reward=3
        elif n_state[0]<state[0] and action==0:
            reward=3
        else:
            reward=-1

        memory.append((state, action, reward, done, n_state))
        
        if len(memory) > batch:
            minibatch=random.sample(memory, batch)

            states=np.array([x[0] for x in minibatch])
            actions=np.array([x[1] for x in minibatch])
            rewards=np.array([x[2] for x in minibatch])
            dones=np.array([x[3] for x in minibatch])
            n_states=np.array([x[4] for x in minibatch])

            target_y=model.call(states).numpy()

            for k in range(batch):
                target_y[k][actions]= rewards+(1-dones)*0.95*np.max(model.call(n_states).numpy(), axis=1)
            
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
            gradients=tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        state=n_state

        if done:
            count+=1
            break

    env.close()    
    print(i+1, t, state[0])

print(count)
```

1. 처음 시도에서는 ```class DQN(Model)``` 부분도 없었고, 보상은 Gymnasium 사이트의 설명 그대로 ```Reward=100 if done else 1```과 같은 형태로 시도했는데, 속도도 느렸고 학습도 제대로 되지 않았음.

2. 두 번쨰에서는 보상 부여가 잘못되었다는 것을 인지하고,

     ```python
        if done :  # reward를 높이에 따라 부여하기
            reward=1000
        elif state[1]>0 and action==2:
            reward=(10*(n_state[0]+0.5)**2 + 5*n_state[1])
        else:
            reward=10*(n_state[0]+0.5)**2
    ```
    과 같은 형태로 부여했으나, 문제가 해결 되지 않았다.

3. 이후 속도 개선을 위해 ```class DQN(Model)```을 사용하고, 보상의 경우에도 조금 더 단순화 하기로 했다. 또한 epi=1000에서 성공률이 그리 높지 않은 것 같아 5000번 실행 후 결과를 살펴보기로 했다.
    ```python
        if done:
            reward=10
        elif n_state[0]>=state[0] and action==2:
            reward=1
        elif n_state[0]<state[0] and action==0:
            reward=1
        else:
            reward=-1
    ```

    ```python
        import matplotlib.pyplot as plt

        counting=[]
        step=[]
        with open("result.txt",'r') as f:
            counter=0
            while True:
                line=f.readline()
                if not line:
                    break
                line=line.split()
                step.append(line[0])
                if float(line[2])>0.5:
                    counter+=1
                counting.append(counter)
        
        fig=plt.figure(figsize=(20,18))
        ax1=fig.add_subplot(1,1,1)
        ax1.plot(step, counting,'b-')
        ax1.set_xticks([0, 999, 1999, 2999, 3999, 4999])
        plt.show()
    ```
    ![result](output.png)
    5000번의 epi에서 성공 횟수가 1400 정도로 성공횟수가 많지는 않았고, 연속된 성공 횟수도 높지 않아 학습이 되지 않는다고 생각했다.

4. memory와 batch의 크기를 키워서 반복해보기로 했다. 뭔가 잘되는 거 같은데...?

## 결론
나는 왜이렇게 멍청한가...
