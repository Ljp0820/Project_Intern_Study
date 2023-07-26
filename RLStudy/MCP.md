Moving CartPole
====
## 서론
1. CartPole을 성공적으로 학습하고 받은 다음 과제로, 기본적으로 CartPole과 똑같은 목표(막대를 넘어뜨리지 않는다.)에 추가적으로 좌 우로 Cart를 이동시키는 목표를 추가적으로 수행해야한다.
2. 처음에는 Reward를 좌나 우로 이동하며 막대를 넘어뜨리지 않을 경우에 Reward를 주는 방향으로 학습을 해 보았으나, 잘 되지 않았다.
3. 더 간단하게 생각을 해 보았는데, 이미 잘 학습된 CartPole Model의 가중치가 존재하는데 처음부터 학습을 시킬 이유가 있나? 라는 생각이 있어, 가중치를 Load하고 그것을 바탕으로 추가적으로 학습을 시키자는 생각이 들었다.
4. 먼저 한쪽 방향으로만 움직이는 학습을 수행했다.<br>
![image](MCP_Inference.gif)<br>
특정 방향으로만 움직이는 결과는 위와 같이 잘 나왔다.
5. 그러나 가중치를 Load 해서 사용하니 중간에 학습이 터져버리는 사고가 매우 빈번하게 일어나서 Load하지 않고 학습하는 방향으로 다시 시도하였으나 결국 방향 한 번에서 두 번 정도로 바꾸는 경우가 최고였다.
![image](MCP1.gif)<br>
## 본론
```python
import numpy as np
import tensorflow as tf
from keras import layers, Model
import random
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode = 'human')

class DQN(Model):

    def __init__(self, state_dim, action_space, learning_r):

        super(DQN, self).__init__()
        self.d1 = layers.Dense(32, activation='relu', input_dim=state_dim)
        self.d2 = layers.Dense(16, activation='relu')
        self.d3 = layers.Dense(16, activation='relu')
        self.d4 = layers.Dense(action_space, activation='linear')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_r)

    def call(self, x):

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        return self.d4(x)
    
   
class Agent():

    def __init__(self):

        self.state_size = 4
        self.action_size = 2
        self.eps = 1.0
        self.eps_decay = 0.98
        self.eps_min = 0.1
        self.batch_size = 32
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.memory = []
        self.model = DQN(self.state_size, self.action_size, self.learning_rate)

    def update_eps(self):

        self.eps = max(self.eps_decay*self.eps, self.eps_min)

    def get_action(self, state):

        if np.random.rand() < self.eps:
            return np.random.randint(0,2)
        else:
            return np.argmax(self.model.call(np.array([state])))
        
    def update_memory(self, n_state, reward, termi, action, state):

        self.memory.append((n_state, reward, termi, action, state))

    def train_model(self):

        if len(self.memory) < 1000:
            return

        if len(self.memory) > 20000:
            del self.memory[0]

        if len(self.memory) > self.batch_size:

            mini_batch = random.sample(self.memory, self.batch_size)

            n_states = np.array([x[0] for x in mini_batch])
            rewards = np.array([x[1] for x in mini_batch])
            termis = np.array([x[2] for x in mini_batch])
            actions = np.array([x[3] for x in mini_batch])
            states = np.array([x[4] for x in mini_batch])

            q_val = self.model.call(states).numpy()
            n_q_val = self.model.call(n_states).numpy()

            targets=q_val.copy()
            targets[np.arange(len(rewards)), actions] = rewards + self.discount_factor * np.max(n_q_val, axis=1) * (1-termis)

            with tf.GradientTape() as tape:
                q_val = self.model.call(states)
                loss = tf.keras.losses.mse(targets, q_val)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

epi = 1000
agent = Agent()
step_memory=[]

for i in range(epi):

    state, _ = env.reset()
    agent.update_eps()
    termi = False
    step = 0
    pos = state[0]
    count=0

    if state[0]<0:
        direction = 'left'
    else:
        direction = 'right'

    while not termi and step<2000:

        act = agent.get_action(state)
        n_state, reward, termi, _, _ = env.step(act)

        if abs(n_state[2]) < 0.209:
            reward = 1
            if direction == 'right':
                if n_state[0] > pos:
                    reward += 2
                    pos = n_state[0]
                else:
                    reward += 1
            elif direction == 'left':
                if n_state[0] < pos:
                    reward += 2
                    pos = n_state[0]
                else:
                    reward += 1
        else:
            reward = -1
            termi = True

        if n_state[0] > 1.8:
            direction = 'left'
        elif n_state[0] < -1.8:
            direction = 'right'

        if abs(n_state[0]) > 2.4:
            termi = True        
        
        agent.update_memory(n_state, reward, termi, act, state)
        agent.train_model()

        step+=1
        state = n_state
    
    step_memory.append(step)
    print("Episode : {:4d} | Step : {:4d}".format(i+1, step))

env.close()
agent.model.save_weights('./save_model_MovingCartPole_1/model', save_format='tf')
plt.plot([j for j in range(len(step_memory))], step_memory)
plt.show()
```

결국 실패했기에 크게 의미가 없는 코드가 되어버렸지만, 보상 구조에 대해 이야기해보자면, <br>먼저 ```env.reset()```을 통해 ```state```를 받고, ```state[0]```에 따라 처음에 어느 방향으로 움직여야 할 지를 정했다. 음수인 경우 왼쪽, 양수인 경우 오른쪽으로 방향을 정하고, 그 방향으로 움직이는 경우에 높은 부상을 부여했다.<br>
```n_state[0] > 1.8```이거나 ```n_state[0] < -1.8```인 경우 방향을 전환하게 했다.
<br>```step```도 원래는 500이지만 2000으로 늘려서 학습 해 봤는데, 2000까지 가는 경우는 이동보다는 Pole을 유지하는 방향으로 학습이 되었고, 1000정도 가는 경우는 한번 방향 전환 하는 경우가 대부분이었다.
## 결론
생각 이상으로 난이도가 높았는데, 문제가 되는 부분이 여러가지가 있었다.<br>
첫 번째는, 보상의 설계이다. 다들 비슷한 고민들을 많이 했을 것이라 생각이 든다. 위에서 보상 설계에 대한 설명을 했지만, 이것이 보상이 잘못된건지, 다른 부분에서 잘못된건지 잘 모르겠다는 생각이 든다. 물론 보상이 잘못되었을 확률이 높긴하지만.<br>
두 번째는, 학습의 종료 조건을 정하는 것이다. 이전의 Pendulum, Cartpole, MountainCar 같은 문제는 정확하고 명시적인 Episode의 종료 조건이 있었다. 그러나 이 Moving CartPole의 경우에는 방향 전환을 하면서 ```step```을 많이 가져가는 것이 목적이어서 이것을 코드화 시켜서 종료 조건을 거는 것이 힘들어 Render를 켜 놓고 게속 관찰하는 것이 상당이 불편했다.<br>
마지막으로 계속된 실패로 인해 의욕이 상실되는 것이...