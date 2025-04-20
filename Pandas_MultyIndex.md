## MultyIndex
```python
# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
```

```python
pop['California'] # California만
pop.loc['California':'New York'] # California 부터 New York 사이 모든 Column
pop[['California', 'Texas']] # cal, tex만
```

```python
health_data['Guido', 'HR'] # guido, hr
health_data.loc[:, ('Bob', 'HR')].unstack() # 모든 인덱스에 대해서 bob의 hr
health_data['Bob', 'HR'].unstack() # bob의 hr의 모든

idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']] # 모든 인덱스의 1번 인덱스, 모든 column 중 HR column
```

```python
data.sort_index() # 인덱스 정렬
pop.reset_index(name='population') # 인덱스 초기화
pop_flat.set_index(['state', 'year']) # 인덱스 재정의
```

---

<br>

## Concat
```python
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])

x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index  # make duplicate indices!

pd.concat([x, y])
pd.concat([x, y], ignore_index=True) # 인덱스 무시
pd.concat([x, y], keys=["x", "y"]) # 멀티 인덱스 키 추가
```

### join
```python
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
pd.concat([df5, df6]) # 빈 칸에 대해선 NaN으로 채움

pd.concat([df5, df6], join="inner") # 교집합
pd.concat([df5, df6], join="outer") # 합집합 (=concat)
```

---

<br>

## Join & Merge
```python
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})

df3 = pd.merge(df1, df2) # concat과는 다르게, 공통된 값으로 병합

pd.merge(df1, df2, on='employee') # key를 주어 기준으로 병합

pd.merge(df1, df3, left_on="employee", right_on="name") # 왼쪽에 employee, 오른쪽에 name이 오도록 병합

pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1) # 병합 후 drop
```

### Set Index
```python
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')

pd.merge(df1a, df2a, left_index=True, right_index=True)
df1a.join(df2a)
```
set_index를 통해 employee라는 공통 인덱스가 생김. 인덱스 기준으로 merge한 결과와 join한 결과는 같음

merge와 inner join의 결과는 같다.

```python
pd.merge(df6, df7, how='outer')

pd.merge(df6, df7, how='left') # left table(df6)을 기준으로 merge
```

```python
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
pd.merge(df8, df9, on="name") # Column의 이름이 같을 경우, _x, _y로 구분
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]) # suffixes로 구분자 명시
```

---

<br>

## Aggregation Grouping
    count() # 합
    first(), last() # 처음, 끝 원소
    mean(), median() # 중앙값
    min(), max() # 최소, 최대
    std(), var() # 표준편차, 분산
    mad() # 평균 절대 편차
    prod() # 곱
    sum() # 합

### Groupby
```python
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['key', 'data1', 'data2'])

df.groupby('key') # groupby object (split 단계)
df.groupby('key').sum() # apply 후 combine

planets.groupby('method')['orbital_period'] # method로 groupby 후 orbital_period에 대해서 aggregate
planets.groupby('method')['orbital_period'].sum()

df.groupby('key').aggregate(['min', np.median, max])

df.groupby('key').transform(lambda x: x - x.mean())

L = [0, 1, 0, 1, 2, 0]
df.groupby(L).sum() # L의 원소가 groupby의 Key가 됨
df.groupby(df['key']).sum()
```

```python
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'} # B와 C는 같은 그룹으로 묶임

df2.groupby(mapping).sum() # key를 인덱스로 지정 후, mapping에 의해 groupby, 후 sum

df2.groupby(str.lower).mean() # key의 대소문자 구분 없이 그룹화 후 mean

df2.groupby([str.lower, mapping]).mean() # mapping과 str.lower()를 동시에 사용하여 그룹화
```

---

<br>

## Pivot Table
```python
titanic.pivot_table('survived', index = 'sex', columns = 'class') # 값에 대해서, index와 column 지정

age = pd.cut(titanic['age'], [0, 18, 80]) # cut : 구간 나누기
titanic.pivot_table('survived', ['sex', age], 'class')
# age의 구간을 나누어 sex와 age를 index로 사용

fare = pd.qcut(titanic['fare'], 2) # qcut : 분위 나누기
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

# 추가적인 옵션
titanic.pivot_table(index = 'sex', columns = 'class', aggfunc = {'survived' : sum, 'fare' : 'mean'})

titanic.pivot_table('survived', index = 'sex', columns = 'class', margins = True)
```