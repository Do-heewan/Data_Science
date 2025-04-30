# Pandas 정리

<br>

## Pandas Series

```python
# Series 객체 생성
data = pd.Series([0.25, 0.5, 0.75, 1.0])

# 인덱스 지정
data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
```

### Dict 구조로 Series 직렬화

```python
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
```

### Series 객체 생성

```python
dct = {2: 'a', 1: 'b', 3: 'c'}
index = [3, 2]

pd.Series(dct)
pd.Series(dct, index=index)
```

---

<br>

## Pandas DataFrame
```python
# dict 구조 정의
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
area_dict = {'California': 423967, 
             'Texas': 695662, 
             'New York': 141297,
             'Florida': 170312, 
             'Illinois': 149995}

# dict 구조 Series 직렬화
population = pd.Series(population_dict)
area = pd.Series(area_dict)

# DataFrame 생성
states = pd.DataFrame({'population': population,
                       'area': area})
```

### DataFrame 구축
```python
# Series 객체를 이용한 방법
pd.DataFrame(population, columns = ['population'])

# dicts를 이용한 방법
data = [{'a' : i, 'b' : 2 * i} for i in range(3)]
pd.DataFrame(data)

# NaN 값
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

# Series 객체를 dicts로 묶음
pd.DataFrame({'population': population, 'area': area})

# 2차원 Numpy배열
pd.DataFrame(np.random.rand(3, 2), columns = ['foo', 'bar'], index = ['a', 'b', 'c'])
```

---

<br>

## Pandas Index 객체

### Immutable Array에서 Index

```python
ind = pd.Index([2, 3, 5, 7, 11])

ind[1] # 3

ind[1] = 2 # Error : Immutable

ind[::2] # Index([2, 5, 11], dtype='int64')

ind.size # 5
ind.shape # (5, )
ind.ndim # 1
ind.dtype # int64
```

### 순서가 있는 Set에서 Index

```python
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])

indA.intersection(indB) # Index([3, 5, 7], dtype='int64')

indA.union(indB) # Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')

# Union - Intersection
indA.symmetric_difference(indB) # Index([1, 2, 9, 11], dtype='int64')
```

---

<br>

## Data Selection

### Series 에서

```python
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])

data['b'] # 0.5

data['a':'c'] # data[a], data[b], data[c] 출력
data[0:2] # data[0], data[1] 출력
data[(data > 0.3) & (data < 0.8)] # 0.3보다 크고 0.8보다 작은 값 (Masking)
data[['a', 'e']] # a, e 값만 (Fancy Indexing)
```

### `loc`, `iloc`, `ix`

```python
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])

data[1] # 'a'
# 1번째 인덱스가 아닌, explicit index 1번으로 해석해야한다. -> explicit index가 우선
data[1:3] # b, c
# 하지만 slicing은 implicit index로 해석된다. 1번 인덱스 부터 3번 인덱스가 아닌, 1번부터 3번까지의 implicit index로 해석된다.
```
>따라서 Implicit인지 Explicit인지 명시적으로 넣어주도록 하여 혼란을 막도록 한다. -> loc, iloc

    1번 인덱스(data.loc[1]) : a, 
    1번째 인덱스(data.iloc[1]) : b

    data.loc[1:3] :     # index[1, 3]에 해당하는 value
    1    a
    3    b
    dtype: object

    data.iloc[1:3] :    # ['a', 'b', 'c'] 의 1 ~ 3 인덱스
    3    b
    5    c
    dtype: object

<br>

### DataFrame에서

```python
area = pd.Series({'California': 423967, 'Texas': 695662, 'New York': 141297, 'Florida': 170312, 'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193, 'New York': 19651127, 'Florida': 19552860, 'Illinois': 12882135})
data = pd.DataFrame({'area': area, 'pop': pop})

data.area is data['area'] # True

data.pop is data['pop'] # pop은 키워드이기 때문에 False반환

data['density'] = data['pop'] / data['area'] # density라는 새로운 행 삽입

data.T # 전치
```

### `loc`, `iloc`, `ix`

```python
data.iloc[:3, :2] # 0 ~ 2행, 0 ~ 1열

data.loc[data.density > 100, ['pop', 'density']] # density가 100보다 큰 값의 pop, density
```

---

<br>

## Ufunction
### Series 및 DataFrame 연산산
```python
ser = pd.Series([6, 3, 7, 4])
np.exp(ser)

df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])
np.sin(df * np.pi / 4)
```

```python
area = pd.Series({'Alaska': 1723337, 'Texas': 695662, 'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193, 'New York': 19651127}, name='population')
population / area
```
Alaska의 population과 New York의 Area 값이 없어 NaN으로 치환

```python
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B
```
마찬가지로 인덱스 0과 3에 대해서는 겹치지 않으므로 NaN (NaN, 5, 9, NaN)

```python
A.add(B, fill_value=0)
```
.add Ufunc을 이용해 연산 가능, fill_value로 없는 값은 0으로 계산 (2, 5, 9, 5)

```python
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list('AB'))
B = pd.DataFrame(rng.randint(0, 10, (3, 3)), columns=list('BAC'))

A + B

fill = A.stack().mean() # mean of all values in A
A.add(B, fill_value=fill)
```

A는 (2, 2) B는 (3, 3)의 데이터 프레임들이다. + 오퍼레이터를 통해 연산을 하면 비어있는 부분은 NaN이 되지만, add연산을 통해 fill_value를 설정하면 비어있는 부분에 fill이 채워지면서 add 연산이 가능하게 된다.

### DataFrame과 Series 간의 연산
```python
A = rng.randint(10, size=(3, 4))
A - A[0]
A - A.iloc[0]
```
A의 모든 열은 A[0]과 연산
(A[0] - A[0], A[1] - A[0], A[2] - A[0])

```python
df = pd.DataFrame(A, columns=list('QRST'))
df.subtract(df['R'], axis=0) # axis=0을 통해 broadcasting 및 연산
df.iloc[0, ::2] # 0번째 인덱스, 처음:끝:2칸씩
# => Q 3 / S 2

df - halfrow # Q행과 S행만 연산, 나머지 NaN
```

---

<br>

## 결측치 처리
```python
vals1 = np.array([1, None, 3, 4])
vals1.sum() # Error : None에 대해선 에러
```

### NaN
```python
vals2 = np.array([1, np.nan, 3, 4]) 

1 + np.nan # NaN
0 *  np.nan # NaN
vals2.sum(), vals2.min(), vals2.max() # NaN

np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2) # NaN을 무시하고 계산
```

```python
pd.Series([1, np.nan, 2, None]) # Series에서 None은 NaN으로 해석
```

### Null값 연산
```python
data = pd.Series([1, np.nan, 'hello', None])

# Null 탐지
data.isnull() # False, True, False, True
data[data.notnull()] # [1, 'hello']

# Null 드랍
data.dropna() # [1, 'hello']

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df.dropna() # nan이 존재하는 row(index) 삭제
df.dropna(axis='columns') # column방향으로 Null값 드랍
df.dropna(axis='columns', how='all') # 모두 Null인 column 드랍
df.dropna(axis='rows', thresh=3) # 최소 thresh개의 non-null값이 있는 row만 남김

# Null 채우기
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))

data.fillna(0) # 1, 0, 2, 0, 3
```