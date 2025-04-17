# Numpy 정리

<br>

## Numpy Array

### 행열 변환

`np.arange(1, 10).reshape(3, 3)` : 1x9를 3x3으로 변환

`x[np.newaxis, :]` : 행 증가

`x[:, np.newaxis]` : 열 증가

### Concate

`np.concatenate([x, y])` : x, y 결합 x와 y는 np.array()

`np.vstack([x, y])` : vertical 방향으로 결합

`np.hstack([x, y])` : horizontal 방향으로 결합

### Split

`np.split([x, [a, b]])` : x를 a에서 한번, b에서 한번 나눔

`np.vsplit([grid, [2]])` : grid를 위 아래로 2개로 나눔

`np.hsplit([grid], [2])` : grid를 왼, 오로 나눔

---

<br>

## Numpy Ufunction

`%timeit` : 동작 시간

|Operator|Ufunction|Description|
|-----|---------|------------|
|+|np.add|1 + 1 = 2|
|-|np.substract|3 - 2 = 1|
|-|np.negative|-2|
|*|np.multiply|2 * 3 = 6|
|/|np.divide|3 / 2 = 1.5|
|//|np.floor_divide|3 // 2 = 1|
|**|np.power|2 ** 3 = 8|
|%|np.mod|9 % 4 = 1|

### 절댓값

`abs(x)` / `np.absolute(x)` / `np.abs(x)`

### 지수 로그

`e^x = np.exp(x)`

`2^x = np.exp2(x)`

`3^x = np.power(3., x)`

`ln(x) = np.log(x)`

`log2(x) = np.log2(x)`

`log10(x) = np.log10(x)`

---

<br>

## Aggregates

`np.sum()` / `np.nansum()` = 원소의 합(sum)

`np.prod()` / `np.nanprod()` = 원소의 곱(product)

`np.mean()` / `np.nanmean()` = 원소의 평균(mean)

`np.std()` / `np.nanstd()` = 원소의 표준 편차(standart deviation)

`np.var()` / `np.nanvar()` = 원소의 분산(variance)

`np.min(), np.max()` / `np.nanmin(), np.nanmax()` = 원소의 최대,최소

`np.argmin()` / `np.argmax()` = 최소값의 index

`np.median()` / `np.nanmedian()` = 원소의 중앙값

`np.percentile()` / `np.nanpercentile()` = 백분위 지수

`np.any()` = 어느 하나라도 True면 True

`np.all()` = 모두가 True면 True

---

<br>

## Broadcasting

![Broadcastiong](https://github.com/user-attachments/assets/ca29c851-a3b0-4f99-a021-ea760bbe227d)

### Rule

- 두 배열의 차원이 다를 때, 차원이 작은 배열의 모양이 바뀐다.
    (2, 3) + (3) => (2, 3) + (2, 3) 연산
- 두 배열의 모양이 어느 하나도 같지 않을 때, 모양이 1인 배열을 같은 모양이 되도록 늘린다.
    (3, 1) + (3) => (3, 1) + (3, 1) 연산
- 어떤 차원에서든 둘 다 크기가 일치하지 않고, 모양이 1이 아닐면 오류 발생.
    (3, 2) + (3, ) => 오류

---

<br>

## Comparison, Mask, Boolean

### Comparison

`==` / `np.equal` : 같으면 True 반환

`<` / `np.less` : 작으면 True 반환

`>` / `np.greater` : 크면 True 반환

`!=` / `np.not_equal` : 같지 않으면 True 반환

`<=` / `np.less_equal` : 작거나 같으면 True 반환

`>=` / `np.greater_equal` : 크거나 같으면 True 반환

<br>

### Boolean Array에서 작업 : entries 개수 세기

`np.count_nonzero(x < 6)` : 6보다 작은 원소 중 0이 아닌 것 개수

`np.sum(x < 6)`

<br>

### any, all

`np.any(x > 8)` : x중 하나라도 8보다 크다면 True 반환

`np.all(x < 10)` : x 원소 모두 10보다 작으면 True 반환

<br>

### Bitwise Operator

`&` / `np.bitwise_and`

`|` / `np.bitwise_or`

`^` / `np.bitwise_xor`

`~` / `np.bitwise_not`

<br>

### Boolean Masks

`x[x < 5>]` : x 중 5보다 작은 원소들만 보임

<br>

### and/or 와 &/|의 차이

&/|는 bit 연산

and/or는 논리 연산

---

<br>

## Fancy Indexing

### [Fancy Indexing](https://github.com/Do-heewan/Data_Science/blob/main/3%EC%A3%BC%EC%B0%A8/ch02-7_fancy_indexing.ipynb)

---

<br>

## Sorting Array

`np.sort()` : 오름차순으로 정렬

`np.argsort()` : 정렬된 배열의 인덱스 출력

`np.sort(X, axis = 1)` : axis 방향으로 X 정렬

### Partial Sort

`np.partition(x, 3)` : 정렬 후 인덱스 3인 원소(4번째로 작은 원소)가 인덱스 3에 고정된 후 왼, 오는 정렬되지 않는 정렬
``` python
x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)

# array([2, 1, 3, 4, 6, 5, 7]) # 4가 인덱스 3에 고정, 왼쪽엔 1, 2, 3 / 오른쪽엔 5, 6, 7이 무작위로
```