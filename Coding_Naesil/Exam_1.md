코딩내실 과제
=====

## 1. 실생활과 코딩
### Q : 회문 검사

회문을 검사하는 문제입니다. 회문이란 앞으로 읽으나 반대로 읽으나 똑같은 문자열을 뜻하는데, 이것을 검사하는 코드입니다.

```python
def check_string(str):

    if len(str)%2 >0:
        ans = False
        return ans
    
    for i in range((len(str))//2):
        if str.pop(0) == str.pop(len(str)-1):
            ans = True
        else:
            ans = False
            return ans
    
    return ans
        
a = "abcdeedcba"
a=list(a)

print(check_string(a))
```

Python의 경우 문자열을 선언하고, 그 문자열을 list로 변환하여 list의 pop method를 사용했습니다. ```s.pop(idx)```를 사용할 경우 idx의 요소를 삭제하고, 그 요소를 return하기 때문에 따로 변수를 선언하지 않아도 바로 그 값에 대한 비교나 연산과 같은 작업을 수행가능합니다.<br>

```cpp
#include <iostream>
#include <vector>
using namespace std;

int check_string(vector<string> str)
{
    string ans="TRUE";
    int size = str.size();

    if (size%2 > 0){
        cout << "FALSE" << endl;
        return 0;
    }
    else{
        size = size/2;
    }

    for (int i=0; i<size; i++){
        if (str.front() == str.back()){
            str.erase(str.begin());
            str.erase(str.begin()+str.size());
        }
        else{
            ans = "FALSE";
            str.erase(str.begin());
            str.erase(str.begin()+str.size());
        }
    }

    cout << ans << endl;

    return 0;
}

int main()
{
    vector<string> v = {"a", "b", "c", "c", "b", "a"};
    check_string(v);
    return 0;
}
```
C++의 경우에는 Vector를 사용했습니다. Python에서 짠 코드를 C++로 구현한 것이니 이해하기는 쉬울거라 믿습니다. C++에는 Python 처럼 자료형 변환이 가능하지 않습니다. 그래서 Vector에 문자열의 요소들을 담고, 그것을 ```check_stirng(v)```이라는 함수에 매개변수로 삽입하여 처리했습니다.<br>
C++에서는 pop을 사용하더라도, return이 없는것으로 알고있어서 ```vector.front()```와 ```vector.back()```과 같은 method로 값을 비교하고, 그것을 지워주는 ```vector.erase(idx)```를 사용해야 합니다. 이 때 idx의 경우 ```vector.begin()```을 시작점을 정해서 그것에 정수를 더해가며 인덱싱을 해야합니다.

## 2. What If

### Q : 배열의 길이가 10인 정수형 array가 있을 때, 배열의 범위 밖인 인덱스 11에서 3 값을 저장하고 그 값을 출력하는 코드를 작성하면 어떻게 될까?
예상: 일단 C++에서는 절대 안될거 같고, Python에서는 알아서 배열의 크기를 늘리든 해서 잘 되지 않을까 하는 막연한 기대가 들었다.<br><br>
결과:
```cpp
#include <iostream>
#include <vector>
using namespace std;

int main()
{
    int A[10];
    int a;

    A[11] = 3;

    cout << 3;

    cin >> a;

    return 0;
}
```

```python
a = [i for i in range(10)]
a[11] = 3
print(a)
```

결과적으로는 Python이나 C++이나 둘 다 List의 Range를 벗어날 경우 Error가 발생했다. 아마도 자료형의 Range를 벗어나서 요소를 추가하는 경우에는 ```a.append()```또는 ```a.pushback()```과 같은 Method를 사용하기에 이렇게 주먹구구식으로 추가하는 경우는 받아들이지 않는 것 같다.

### Q : 자료형 변환
#### Python과 C++에서의 자료형 변환은 어떻게 할까? 가능할까?

예상 : 일단 Python에서는 자료형 변환이 아주 쉽게 일어납니다. 내장함수를 사용하여 ```list(string)```과 같은 방법으로 문자열을 쉽게 list로 변환 가능하고 ```int(float num)```과 같이 Float를 Int로 변환도 가능합니다. 그런데 C++에서도 이런 내장함수가 있어서 변환이 될까요? 아무래도 불가능할 것 같습니다.

```python
pi = 3.141592
string = "Hello"
a = [1,2,3,4,5]

print(int(pi))
print(list(string))
print(set(string))
print(tuple(a))
```
```
3
['H', 'e', 'l', 'l', 'o']
{'e', 'H', 'l', 'o'}
(1, 2, 3, 4, 5)
```
위 결과와 같이 아주 쉽게 실수가 정수로, 문자열이 list에 들어갑니다. 또한 list에서 tuple과 같은 Container 자료형끼리의 형 변환도 쉽게 가능합니다.<br>
추가적으로 ```set(string)```을 보면 Python에서 set을 사용하는 주 목적을 알 수 있는데, 바로 중복된 요소 제거입니다. string문자열의 중복된 요소, 'l'을 제거하여 하나만 남깁니다.

```cpp
#include <iostream>

int main(){

    float pi = 3.141592;
    std::cout << int(pi) << std::endl;

    return 0;
}
```
솔직히 c++에서 float를 int로 바꾸는 것도 안될 줄 알았는데, 되네요?<br>
찾아보니 Static_cast, Dynamic_cast, const_cast, reinterfret_cast와 같은 형변환이 있는데 그만 정신이 아득해져서...
```cpp
#include <vector>
#include <list>

int main(){

    std::vector<int> v = {1,2,3,4};
    std::list<int> a;

    a=v;

    return 0;
}
```
뭐 당연히 오류로 안됩니다...<br>
반복문을 활용하여 요소를 하나 하나 꺼내서 다른 자료형에 넣어주는 방법으로 자료형 변환이 될 거 같은데, 이렇게 하는거도 자료형 변환이라고 부를 수 있을지...<br>

결론 : python에서는 내장함수로 손쉽게 자료형 변환이 되지만, C++에는 실수를 정수로 변환 가능하지만, container 자료형을 변환하는 내장함수는 없다.
