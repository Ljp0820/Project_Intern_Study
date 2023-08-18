함수의 사용
====
## What If
### 1. 함수에서 여러 값을 반환 하는 법 (C++)
Python에서는 함수의 반환 자료형을 미리 지정하지 않아도 되기에 return을 어떻게 줘도 되기에 여러 값을 반환 하는 경우에 알아서 반환이 됩니다. 예시로는 다음과 같습니다.<br>

```python
def whatif():

    alpha = 10
    beta = 1
    gamma = 23

    return alpha, beta, gamma

print(whatif())
```

당연히 위 코드는 작동할 것입니다. 만약 이렇게 여러 값을 반환 하면 반환 값의 자료형은 무엇일까요? 

C++에서는 여러 값을 반환 하기 위해서는 어떻게 해야할까요? <br>한 번 생각해보고 아래를 읽어봅시다. 생각안해보고 내리지 마세요!!!!!!!!!!<br><br><br><br>
처음으로 드는 생각은 함수의 반환 값을 Vector와 같은 Container로 지정하면 될 거 같습니다.
```cpp
#include <vector>
using namespace std;

vector<int> func(){

    vector<int> returnVec = {0,0,0,0,0};

    return returnVec;

}
int main(){

    vector<int> Vec;

    Vec = func();

    return 0;

}
```

디버거로 Vec의 값이 어떻게 변하는 지 관찰해봅시다.<br>

또 다른 방법으로는 매개변수에 포인터나 얕은복사를 사용하는 방법이 있습니다.<br>

```cpp
using namespace std;
void func(int* a, int* b, int& c, int &d){

    *a += 10;
    b += 11;
    c += 12;
    d += 13; 

}

int main(){

    int a=0, b=0, c=0, d=0;

    func(&a, &b, c, d);

    return 0;

}
```
디버거를 사용하여 각 변수들이 어떻게 변하는지 살펴봅시다.<br>
a와 b의 경우 포인터를 이용했고, c와 d는 얕은 복사를 이용했습니다.<br>
두 방법 모두 변수의 주소를 이용해 main함수의 변수를 변경하는 방법으로 값을 여러개 반환하는 것 처럼 사용가능합니다.<br>

마지막 방법으로는 ```std::pair<자료형, 자료형>```을 이용하는 방법이 있는데, 다음과 같다.<br>

```cpp
using namespace std;

pair<int, string>Func(int a, string b){

    return make_pair<int, string>(a+10, b+"world");

}

int main(){

    Func(10, "Hello");

    return 0;

}
```
3가지 방법 이외에 방법이 있다면 찾아보고, 어떤 방법을 주로 사용할 것인지 고민해보는 시간을 가집시다.

### 2. 재귀함수의 사용
재귀함수를 앞선 시간에 배웠는데, 사실 재귀함수는 함수 사용 횟수가 정해진 경우에 사용하는 것이 좋습니다. 더 솔직하게 말하자면, 재귀함수를 쓰는 일이 잘 없기도 합니다. 주로 반복문을 이용하는 것이 재귀함수를 사용하는 것 보다 편하기 때문입니다.<br>

재귀함수를 사용하는 경우에는 이전 스터디에서도 언급 되었듯, 기저 조건을 걸어줘야하는데, 그 기저 조건에 대한 생각을 코드로 표헌하는 것이 쉽지는 않을 것입니다. 추가적으로 재귀함수의 가장 큰 단점이 하나 있는데, 무엇일지 한번 고민해보고 아래로 내려 확인해봅시다.<br>

주로 Python에서 재귀함수를 너무 많이 호출하면 Recursion Error가 발생합니다. 아래 코드를 실행해봅시다.
```python
def factorial(n):

    if n == 1:
        return 1
    
    return n*factorial(n-1)

factorial(10000)
```
실행해보면 결과가 다음과 같습니다.
```
RecursionError: maximum recursion depth exceeded in comparison
```

그렇다면 C++에서는 어떻게 될까요? C++에서는 재귀함수를 많이 사용하면 RecursionError가 발생할까요?<br>

정답은 C++에서도 발생한다입니다.

이와 같은 오류가 발생하는 이유는 디버거를 사용 해보면 알 수  있는데, 위 Python 코드를 디버깅 하고, 디버거 창에 Call Stack을 관찰해보면 facorial이 계속해서 생성되는 것을 알 수 있습니다. Call Stack이라는 말에서 알 수 있듯, 재귀함수를 사용하면, 호출하고 그것을 Stack하기 때문에 자원의 한계(메모리겠죠?)를 넘어갈 경우 Stack을 하지 못하기에 Recursion Error가 발생합니다. 특히나 Python의 경우 원래 무겁고 메모리 사용량이 큰 언어이기에 빨리 Error가 발생할 것이고, C++의 경우에도 언젠가는 메모리가 터지기 때문에 오류가 발생합니다.<br>
그러므로 재귀함수를 쓰는 것을 저는 권장하지 않습니다. 만약 무조건 사용하고 싶다면, 재귀 함수의 호출 횟수가 정해진 상황에서 사용하는 것이 좋을 것입니다. 그런데 호출 횟수가 정해진 경우엔 for문으로, 그렇지 않을 경우 while문으로 돌리면 되는데 굳이 라는 생각이 들긴합니다...

## 문제

먼저 함수를 사용하는 이유에 대해 생각해봅시다. 여러 이유가 있을 것입니다. 코드를 재사용한다던가... 유지보수에 용이하다던가... 메인코드가 보기 좋아진다던가... 제가 이때까지 코드를 짜면서 느꼈던 함수를 썼을 때 가장 편한점은 아무래도 반복문에 조건문으로 탈출 조건을 거는 경우 ```return```을 활용해서 반복문을 몇번을 중첩해도 한번에 탈출 가능하다는 것입니다.<br>

### 문제 설명

올바른 괄호가 사용되었는지 판별하는 함수 ```detect()```를 완성하세요.<br>
올바른 괄호란 반드시 '(' 로 시작되고 짝지어 ')'로 닫혀야합니다.<br>
예를 들어 "))((((알찬 스터디))"는 잘못된 괄호 사용입니다. "(((()((알찬 스터디)))))"는 올바른 괄호 사용입니다.<br>
함수의 매개변수와 반환값의 자료형은 가이드 코드에 명시해놓았습니다.

```python
def detect(s: str) -> (str, list):

   ### Enter Your Code Plz...

is_right, string = detect("(((Happy Study)))")
print("{:s} {:s}".format(is_right, ''.join(string)))
```
예시)
```python
입력 : "((())Happy Study)))"
출력 : 괄호사용이 잘못되었습니다.

입력 : "((((Happy Study))))"
출력 : 올바른 괄호 사용입니다. Happy Study
```

```cpp
#include <iostream>
using namespace std;

int detect(// 매개 변수도 직접 채워봅시다.//
){

    // Enter Your Code

}

int main(){

    string str = "((((Happy Study))))", is_right, words;

    detect(// 매개변수도 직접 채워봅시다.//
    );

    cout << is_right << words << endl;

    return 0;
}
```

```
결과 : Right Parentheses. Happy Study
```

### 정답코드
```cpp
#include <iostream>
using namespace std;

int detect(string s, string* is_true, string* word){

    int count = 0;
    string conv_word;

    for(auto elem : s){
        if(elem == '('){
            count += 1;
        }else if(elem == ')'){
            count -= 1;
            if(count < 0){
                *is_true = "Wrong Parentheses. ";
                *word = "\n";
                return 0;
            }
        }else{
            *word+=elem;
        }
    }
    if(count!=0){
        *is_true = "Wrong Parentheses. ";
        *word = "\n";
        return 0;
    }else{
        *is_true = "Right Parentheses. ";
        return 0;
    }

}

int main(){

    string str = "((((Happy Study))))", is_right, words;

    detect(str, &is_right, &words);

    cout<<is_right<<words<<endl;

    return 0;
}
```
```python
def detect(s: str) -> (str, list):

    is_true = ''
    words = []
    count = 0
    for word in s:
        if word == '(':
            count += 1
        elif word == ')':
            count -= 1
            if count < 0:
                is_true = '괄호사용이 잘못되었습니다.'
                words = []
                return is_true, words
        else:
            words.append(word)
    
    if count != 0:
        is_true = '괄호사용이 잘못되었습니다.'
        words = []
        return is_true, words
    else:
        is_true = '올바른 괄호 사용입니다.'
        return is_true, words

is_right, string = detect("(((Happy Study)))")
print("{:s} {:s}".format(is_right, ''.join(string)))
```
