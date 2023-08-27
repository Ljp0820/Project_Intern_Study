Class 구현
=====
## What If
이번주 what if는 떠오르는게 없어서 날로먹으려고 이때까지 풀어보았던 연습문제, 과제들을 Class로 구현해보겠습니다.<br>
솔직히 과제 수준의 문제에서 class를 상속하려고 사용할 일은 잘 없다고 생각하고요, 개인적으로는 함수를 많이 쓰고, 매개변수를 많이 적기 싫을 때 사용하는거 같습니다.<br>
```__init__```에 ```self.변수```로 다 퉁쳐버리면, 다른 함수에 매개변수로 self만 적어버리면, 모든 변수가 알아서 넘어가니 좋은거 같아요.<br>
또 코드를 이해해야하는 입장에서는 class로 구현된 코드를 이해하기 쉽습니다. 아래 코드를 보면, main함수가 진행되는 부분이 깔끔해지고, 각각의 메서드를 사용할 때, 메서드를 기능별로 끊어서 작성해 놓았기 때문에 그 메서드가 어떤 기능을 하는 지 이해하기도 쉽습니다.<br>
class는 구현해보고, 안해보고 차이가 엄청 크다고 생각해서 자신이 냈던 문제를 본인이 class 형식으로 다시 구현해보는 것 정도는 해보셨으면 좋겠습니다.<br>

### 1. 숫자야구

일단은 여러가지 함수를 사용하는 문제들은 class로 구현하는게 좋다고 생각해요. 주어진 상황에서 추가적으로 기능이 필요하거나 하는 상황이 오면, ```def func_name(self)```로 메서드 하나 더 만들어서 추가만 하면 되니까요.<br>
숫자야구에서 필요한 기능을 생각해보고, 그 기능들을 메서드로 하나 하나 구현하면 됩니다. self 변수는 처음에 필요할만한거 적어놓고, 그떄그때 ```__init__```에 추가만 하면 매개변수를 추가로 적어줄 필요도 없으니 얼마나 좋아요.<br>
아래 코드에서는 secret_num을 생성하는 메서드, guess_num을 입력받는 메서드, check_strike 메서드, 결과를 print하고 while문 탈출을 결정하는 메서드 이렇게 4가지의 메서드를 사용했습니다.<br>
```python
import numpy as np

class Baseball:

    def __init__(self):

        self.secret_num = []
        self.guess_num = 0
        self.strike = 0
        self.ball = 0
        self.out = 0

    def gen_secret_num(self):

        ### secret_num을 생성하는데, 세자리의 숫자 중에서 중복되지 않아야함
        ### 123은 되지만, 112는 안되기에 set을 활용해서 중복없는 세자리수 list에 저장.

        size = 1
        check_num = set(self.secret_num)

        while True: ### 몇번 반복해야하는지 모르니까 while사용
            new_num = np.random.choice(np.arange(1,10))
            check_num.add(new_num)
            if size == len(check_num): #set에 add하고 중복이면 set의 크기가 증가하지 않겠죠?
                self.secret_num.append(str(new_num))
                size+=1
            
            if len(check_num) == 3:
                break

    def get_num(self):

        self.guess_num = int(input("Enter Your Guess : "))
    
    def check_strike(self):

        ### strike와 ball, out의 갯수를 check하는 메서드
        ### 값이 같은것과, 값과 인덱스가 같은것, 둘다 다른것 체크하기

        guess = list(str(self.guess_num))
        self.strike = self.ball = self.out = 0

        for i in range(len(self.secret_num)):
            count = 0
            for j in range(len(guess)):
                if guess[j] == self.secret_num[i]:
                    if j == i:
                        self.strike+=1
                    else:
                        self.ball+=1
                else:
                    count += 1
            if count==3:
                self.out+=1
    
    def result(self):
        ### 결과 출력하기

        print("{:d} Strikes, {:d} Balls, {:d} Outs".format(self.strike, self.ball, self.out))
        if self.strike == 3:    ### 만약 strike가 3이면 더 게임할 필요 없음.
            print("Game Finished")
            return True
        else:
            return False

baseball = Baseball()
flag = False                ### flag로 while문 반복 여부 결정.
baseball.gen_secret_num()   ### secret_num은 한번만 초기화 하면 됨.

while not flag:
    baseball.get_num()
    baseball.check_strike()
    flag = baseball.result()
```

### 2. Sensor Node와 Sink Node

이런 논리를 구현하는 문제는 정말 나중에 sink와 sensor에 기능들을 추가해야하는 상황이 올 수도 있으니, 진짜로 class로 구현할만한 가치가 있다고 생각합니다. <br>
Sensor와 Node의 Class를 만들고, 나중에 기능을 추가한다면 상속받아서 추가하면 될거 같다는 생각이 들지 않나요?<br>

```python
import numpy as np
import matplotlib.pyplot as plt

class Sink:
                            ### Sink class에서는 Sink의 계획 list를 return
    def __init__(self):
        
        self.msat = 20
        self.satr = 20
        self.msst = 10
        self.sstr = 10
        self.list = []

    def make_list(self):

        for i in range(5):
            awake_time = np.random.randint(self.msat - self.satr/2, 1 + self.msat + self.satr/2)
            sleep_time = np.random.randint(self.msst - self.sstr/2, 1 + self.msst + self.sstr/2)
            self.list += [1 for _ in range(awake_time)] + [0 for _ in range(sleep_time)]
        
        return self.list

class Sensor:
                        ### Sensor class 에서는 Sink의 schedule과 time_step을 추가로 input으로 받는다.

    def __init__(self, schedule_list, time_step):

        self.sink_schedule = schedule_list
        self.time_step = time_step
        self.sensor_schedule = []
    
    def make_sensor_schedule(self):

        count = 0

        for i in range(len(self.sink_schedule)-1):
            if self.sink_schedule[i] != self.sink_schedule[i+1]:
                count += 1
                if self.sink_schedule[i] == 0:
                    self.sensor_schedule += [0 for _ in range(count)]
                elif self.sink_schedule[i] == 1:
                    init_time = np.random.randint(0, count+1-self.time_step)
                    self.sensor_schedule += [0 for _ in range(init_time)] + [1 for _ in range(self.time_step)] + [0 for _ in range(count - self.time_step - init_time)]
                    ### python에서는 list에 그냥 덧셈 하면 자동으로 list끼리 합쳐지기 떄문에 이런 한줄 코딩도 가능합니다.
                    ### numpy에서는 더하기 하면 인덱스에 맞게 더해지므로 불가능합니다.
                count = 0
            else:
                count += 1

        self.sensor_schedule += [0 for _ in range(count+1)]
        
        return self.sensor_schedule
            
sink_1 = Sink()
schedule = sink_1.make_list()
sensor_1 = Sensor(schedule, 5)
sensor_schedule = sensor_1.make_sensor_schedule()

plt.plot(schedule, label='Sink')
plt.plot(sensor_schedule, label="Sensor")
plt.legend()
plt.show()
```

### 3. 보드게임

이런 격자판에서 움직이는 문제들은 class로 많이 구현해봤고, class가 더 편한거 같아요.<br>
격자판 위의 위치를 받는 기능, 말을 이동시키는 기능, 이동한 위치를 검사하는 기능, 결과로 격자판을 출력하는 기능과 같이 어떤 기능이 필요한지 다른 문제에 비해 명확하게 드러나는편이라
class로 풀기 쉬운편이라 생각합니다. 거기다 2차원 이상의 Container들을 매개변수로 항상 입력해주지 않아도 된다는점도 장점일거같아요.
```python
class Board_Game:

    def __init__(self):
        
        self.table = [[1,0,0,0,0], [0,0,0,0,0], [0,0,5,-1,0], [-1,0,0,0,0], [0,0,0,0,10]]
        self.before_state = [0, 0]
        self.now_state = [0, 0]
        self.ice = [2, 2]
        self.target = [4, 4]
        self.hole = [[3, 0], [2, 3]]

    def get_act(self):

        self.action, self.distance = map(int, input("Enter your action and distance").split())
    
    def move(self):

        print(self.before_state, self.now_state)

        if self.action == 1:
            self.now_state[0] -= self.distance # up
        elif self.action == 2:
            self.now_state[0] += self.distance # down
        elif self.action == 3:
            self.now_state[1] -= self.distance # left
        elif self.action == 4:
            self.now_state[1] += self.distance # right

        print(self.before_state, self.now_state)

    def check_state(self):

        if (self.now_state[1] > 4) + (self.now_state[1] < 0) + (self.now_state[0] > 4) + (self.now_state[0] < 0):
            self.now_state[0] = self.before_state[0]
            self.now_state[1] = self.before_state[1]
            print("Wall")
            return False
        
        if self.now_state == self.ice:
            self.distance=1
            print("Ice")
            self.move()
            self.table[self.before_state[0]][self.before_state[1]] = 0
            self.table[self.now_state[0]][self.now_state[1]] = 0
            self.table[self.ice[0]][self.ice[1]] = 5
            self.check_state()
            return True
        
        if self.now_state in self.hole:
            self.now_state = [0,0]
            self.before_state = [0,0]
            print("Hole")
            return True
        
        if self.now_state == self.target:
            self.table[self.before_state[0]][self.before_state[1]] = 0
            print("End")
            return False

        return True
        
    def update_table(self):
        self.table[self.before_state[0]][self.before_state[1]] = 0
        self.table[self.now_state[0]][self.now_state[1]] = 1
        self.before_state[0] = self.now_state[0]
        self.before_state[1] = self.now_state[1]

    def print_table(self):
        for list in self.table:
            print(list)
        print("----------------")

game = Board_Game()
flag = True
game.print_table()

while flag:

    game.get_act()
    print(game.action, game.distance)
    game.move()
    flag = game.check_state()
    game.update_table()
    game.print_table()
```

### 4. Poisson 분포와 Exp 분포 수렴확인
저번주의 실수와 같은 문제인데, 사실 처음 문제 낼 때 부터 class로 만들었어요. 상속 받아서 다른 분포에 대한 메서드를 추가하면, Poisson 분포와 Exp 분포 말고도 다른 분포를 비교할 수 있지 않을까요?

```python
import matplotlib.pyplot as plt
import numpy as np

class Verify_distribution:

    def __init__(self):

        self.lam=5
        self.time=5
        self.iter=1000
        self.avg = self.lam*self.time
        self.poisson_avg=[]
        self.exp_avg=[]
    
    def poisson(self):

        poisson_list = []

        for step in range(self.iter):
            poisson_list.append(self.time*np.random.poisson(self.lam))
            self.poisson_avg.append(sum(poisson_list)/(step+1))
        
        return self.poisson_avg

    def exponential(self):

        exp_list = []

        for step in range(self.iter):
            count = 0
            init_time = 0
            while True:
                exp_time = np.random.exponential(1/self.lam)
                init_time += exp_time
                if init_time < 5:
                    count+=1
                else:
                    break
            exp_list.append(count)
            self.exp_avg.append(sum(exp_list)/(step+1))
        
        return self.exp_avg

    def draw_graph(self, first_dis, second_dis):

        plt.plot(first_dis)
        plt.plot(second_dis)
        plt.axhline(self.avg, color='red', linestyle='--', label='Average')
        plt.show()

verify = Verify_distribution()
verify.draw_graph(verify.poisson(), verify.exponential())
```

4문제를 class로 바꿔서 구현해보았는데, 자신이 이전에 풀어봤던 코드와 비교해보면서 class를 왜 사용했는지... 사용하면 좋은거 같은지... 혼자서 생각해보는 시간을 가지시면 좋겠어요.<br>
추가적으로 다른 문제도 직접 class로 구현해보면서 class에 익숙해지셨으면 좋겠습니다.

## 문제 1. Class의 상속 사용해보기

whatif의 마지막 문제였던 분포 생성기 class를 상속받아서 추가로 다른 분포를 비교해보는 코드를 작성해보세요. Poisson 분포와 Exp 분포를 비교해봤으니 다른 분포도 수렴하는지 궁금하지 않으신가요?<br>
가장 쉬운 예로는 uniform 분포가 정말 수렴하는가? 정도만 해보셔도 될 거 같습니다.<br>
Guide Code는 위의 코드를 복사해서 사용하시면 될 거 같아요.