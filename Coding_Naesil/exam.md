반복문 연습하기
====
## 생각해보기 (What if)
Container와 연관된 For문은 Container 내부에 인덱싱으로 접근할지, 그 값 자체로 접근할 지에 따라 사용하는 방식이 달라지게 됩니다. 그런데 List와 Vector는 어떻게 사용하는 지 배웠는데 Dictionary와 Map의 경우 For문을 통해 Container 안의 자료에 어떻게 접근할까요?<br>
앞서 배웠듯 ```for i in dictionary```나 ```for(auto elem : Map)```을 사용하면 Dictionary와 Map의 Key와 Value에 접근이 될까요?<br>
관련 정보를 찾아보고, 다음 연습 문제를 풀어봅시다.<br>
추가적으로 두 가지의 for문을 배웠습니다. 인덱싱으로 자료에 접근하는 방법과, 값 자체에 접근하는 방법 두 가지를 배웠는데 어떤 상황에 어떤 접근 방법을 사용해야 할 지도 생각을 해보고 다음 문제로 넘어가봅시다.
## List(Vector)와 Dictionary(Map)에서 반복문 사용하기
member는 dictionary(map)입니다. 사용자가 key, 누적 신고 횟수가 value로 초기화 되어 있습니다.<br>
records는 2차원 list(vector)입니다. 신고 내역에 대한 기록입니다. Python 코드로 예를 들면, ```records[0]```은 은주가 신고한 내역입니다.<br>
records를 바탕으로 신고 횟수를 update하고, 3회 이상 신고될 경우 사용자를 1달 정지시킵니다.<br>
출력으로는 누적 신고횟수와 몇 달 정지인지를 출력하는 코드를 작성해봅시다.(코드의 주석을 참고하세요)

```python
member = {'은주' : 0, '신영' : 0, '재필' : 0, '대헌' : 0}
records = [['신영', '재필', '대헌'], ['은주', '은주', '재필', '대헌', '은주', '은주', '은주'], ['대헌', '대헌', '신영', '은주', '대헌'], ['재필']]

## endter your code

## print문 입니다. 참고하시고~
## print("{:s}님의 누적 신고 횟수는 {:1d}회로 {:1d}달 정지입니다.".format(변수1, 변수2, 변수3))
```

```cpp
// cpp에서는 자료형에 한글 넣으면 이상하게 저장되어 문자열 비교가 안되서 이름을 영어로 바꿨습니다...
// 사용하는 헤더파일도 guide로 드리면 힌트가 되지 않을까 싶습니다.

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
using namespace std;

int main(){

    map<string, int> members = {{"Mike", 0}, {"Nigo", 0}, {"Rei", 0}, {"Yves", 0}};

    vector<vector<string>> records = {{"Nigo", "Rei", "Yves"},
                                     {"Mike", "Mike", "Rei", "Mike", "Yves", "Mike", "Mike"},
                                     {"Nigo", "Yves", "Yves", "Mike", "Yves"},
                                     {"Rei"}};
    
    // enter your code

    // cout문입니다. 참고하세요~
    //cout << elem.first << "님은 신고횟수 " << elem.second <<"회로 " << i << "달 정지입니다." << endl;

    return 0;
}
```

### 나름의 정답
```python
member = {'은주' : 0, '신영' : 0, '재필' : 0, '대헌' : 0}
records = [['신영', '재필', '대헌'], ['은주', '은주', '재필', '대헌', '은주', '은주', '은주'], ['대헌', '대헌', '신영', '은주', '대헌'], ['재필']]

for record in records:
    for value, item in member.items():
        if record.count(value) >= 5:
            record = []
    for who in record:
        if who in member:
            member[who] +=1
            
for value, items in member.items():
    i=0
    if (items >= 3):
        i+=1
    print("{:s}님의 누적 신고 횟수는 {:1d}회로 {:1d}달 정지입니다.".format(value, items, i))
```

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
using namespace std;

int main(){

    map<string, int> members = {{"Mike", 0}, {"Nigo", 0}, {"Rei", 0}, {"Yves", 0}};

    vector<vector<string>> records = {{"Nigo", "Rei", "Yves"},
                                     {"Mike", "Mike", "Rei", "Mike", "Yves", "Mike", "Mike"},
                                     {"Nigo", "Yves", "Yves", "Mike", "Yves"},
                                     {"Rei"}};
    
    int i;

    for(i=0; i<records.size(); i++){
        for (auto member : members){
            if (count(records[i].begin(), records[i].end(), member.first) >= 5){
                records[i] = {};
            }
        }
        for (auto who : records[i]){
            if (members.find(who) != members.end()){
                members[who]++;
                }
        }
    }
    for (auto elem : members){
        i=0;
        if(elem.second >= 3){
            i++;
        }
        cout << elem.first << "님은 신고횟수 " << elem.second <<"회로 " << i << "달 정지입니다." << endl;
    }
    return 0;
}
```