조건문 연습하기!
=======
## Q1. 바둑판에서 말 움직이기
조건문 연습을 위한 문제입니다.<br>
7*7 바둑판에서 내 말을 움직입니다. 상대의 말은 정수 2로 표시되었고, 빈칸은 정수 0, 내 말은 정수 1로 표시되어있습니다.<br>
1회의 공격만 한다고 가정하고, 1회의 공격시 5회 이동가능합니다. Python의 경우 이동은 상, 하, 좌, 우로 C++의 경우 한국어 입력이 인식이 안되서 up, down, left, right로 입력합니다.<br>
내 말을 이동할 때, 빈칸의 경우 내 말을 추가로 놓습니다. 상대말이 있을 경우 말을 잡고 잡았다는 표시로 -2를 입력합니다. 바둑판을 넘어간 경우 이전위치로 돌아갑니다.<br>
처음에는 모든 코드를 작성하게 하려고 했지만, 조건문 문제 연습이니 조건문을 사용해야하는 곳만 작성하면 되게 했습니다.
```python
### Python Code

plate = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0]]
actions = input().split()
print(actions)
row, col = 3, 3

# enter your code

for i in range(len(actions)):

    # enter your code
    
    print("{:6d}번쨰 이동 {:s}".format(i+1, actions[i]))
    for j in range(len(plate)):
        print(plate[j])
    print("=====================")

```
```cpp
/// C++ Code

#include <iostream>
#include <vector>
using namespace std;

int main(){

    vector<vector<int>> plate = {{0, 0, 0, 2, 0, 0, 2},
                                {0, 2, 0, 0, 0, 0, 2},
                                {0, 0, 0, 2, 0, 0, 0},
                                {0, 0, 0, 1, 0, 0, 0},
                                {2, 0, 0, 0, 0, 0, 0},
                                {0, 0, 0, 0, 0, 2, 0},
                                {2, 0, 0, 0, 0, 0, 2}
                                };
    string action;
    int i, j, row=3, col=3;

    for(i=0; i<5; i++){
        cin >> action; // 입력 영어로 해야합니다 {up, down, left, right}

        ///Enter your code

        cout << i+1 << "번째 명령 : " << action << endl;
        for(j=0; j<plate.size(); j++){
            for(int k=0; k<plate[0].size(); k++){
                cout << plate[j][k];
            }
            cout << endl;
        }
        cout << "==============" << endl;
    }

    return 0;
}
```

나름의 정답입니다.
```python
plate = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0]]
actions = input().split()
print(actions)
row, col = 3, 3

if len(actions) != 5:
    print('이동횟수가 부족하거나 많습니다.')

for i in range(len(actions)):
    if actions[i] == '좌':
        col -= 1
        if col < 0:
            col += 1
    elif actions[i] == '우':
        col += 1
        if col > 6:
            col -= 1
    elif actions[i] == '상':
        row -= 1
        if row < 0:
            row += 1
    elif actions[i] == '하':
        row += 1
        if row > 6:
            row -= 1
    
    if plate[row][col] == 2:
        plate[row][col] = -2
    elif plate[row][col] == -2:
        pass
    else:
        plate[row][col] = 1
    
    print("{:6d}번쨰 이동 {:s}".format(i+1, actions[i]))
    for j in range(len(plate)):
        print(plate[j])
    print("=====================")
```
```cpp
#include <iostream>
#include <vector>
using namespace std;

int main(){

    vector<vector<int>> plate = {{0, 0, 0, 2, 0, 0, 2},
                                {0, 2, 0, 0, 0, 0, 2},
                                {0, 0, 0, 2, 0, 0, 0},
                                {0, 0, 0, 1, 0, 0, 0},
                                {2, 0, 0, 0, 0, 0, 0},
                                {0, 0, 0, 0, 0, 2, 0},
                                {2, 0, 0, 0, 0, 0, 2}
                                };
    string action;
    int i, j, row=3, col=3;

    for(i=0; i<5; i++){
        cin >> action;
        if (action.compare("up") == 0){
            row--;
            if(row < 0){
                row++;
            }
        }else if(action.compare("down") == 0){
            row++;
            if(row>6){
                row--;
            }
        }else if(action.compare("left") == 0){
            col--;
            if(col<0){
                col++;
            }
        }else if(action.compare("right") == 0){
            col++;
            if(col>6){
                col--;
            }
        }else{
            cout << action << "은잘못된 이동입니다" <<endl;
            break;
        }

        if(plate[row][col] == 2){
            plate[row][col] = -2;
        }else if(plate[row][col] == -2){
            plate[row][col] = -2;
        }else{
            plate[row][col] = 1;
        }

        cout << i+1 << "번째 명령 : " << action << endl;
        for(j=0; j<plate.size(); j++){
            for(int k=0; k<plate[0].size(); k++){
                cout << plate[j][k];
            }
            cout << endl;
        }
        cout << "==============" << endl;
    }

    return 0;
}
```

## 조건문을 구현하면서 생각해볼점
### 부울대수의 연산은 어디까지 될까요?

python에서 부울대수
```python
True + True #가능합니까? 가능하다면 결과는 어떻게 나올까요?
True + False + 3 #결과는 어떻게 나올까요?
(True or False)*10 # 이런거도 가능할까요?
```
놀랍게도 3가지 연산 모두 결과가 출력됩니다. 역시 파이썬이다~
```python
a=1
b=0
if (a or b) == True:
    print("이게 되네")
#가능할까요??
```
이것 또한 ```(boolean) == (integer)``` 서로 다른 자료형이지만 논리 관계가 성립합니다.

c++에서는 어떻게 될까요?
```cpp
#include <iostream>

int main(){
    bool a=true, b=false;

    if( a||b + 1 == 2){
        std::cout << "이게 되네..." << std::endl;
    }else{
        std::cout << "이게 안되네..." << std::endl;
    }
    return 0;
}   
////결과가 어떻게 될까요?
////a||b 결과는 어떤 자료형일까요? bool? int? float?
```
안될 것 같지만 C++에서도 논리연산을 한 결과가 정수형으로 나오고, 연산이 추가적으로 가능합니다.<br>
자료형은 int에도 들어가고 float에도 들어가네요.
```cpp
int main(){
    bool a=true, b=false;
    /// a+b, a*b (a+b)*10가능한가요? 결과는 어떻게 나올까요?
    /// 자료형은 어떻게 될까요??
}
```
a+b, a*b의 경우는 연산이 되고, (a+b)*10의 경우는 안되네요.
