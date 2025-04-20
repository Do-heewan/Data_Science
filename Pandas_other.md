## Vectorized String
```python
data = ['peter', 'Paul', 'MARY', 'gUIDO']
names = pd.Series(data)

# 첫글자 대문자, 이후 소문자
names.str.capitalize() # pandas Series 객체에선 str 메서드 사용 가능
```

### String Method
```python
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam', 'Eric Idle', 'Michael Palin'])

monte.str.lower() # 모두 소문자
monte.str.len() # 길이
monte.str.startswith('T') # 시작 문자 검색 (True/False)
monte.str.split()
```

### 정규표현식
```python
monte.str.extract('([A-Za-z]+)', expand = False) # 정규 표현식 사용
monte.str.extract('([A-Za-z]+)', expand = True) # expand = True로 DataFrame 형태로 반환
monte.str.findall(r'^[^AEIOU].*[^aeiou]$') # AEIOU로 시작하지 않고, aeiou로 끝나지 않는
```

### other
```python
# Vectorized item and slicing
monte.str[0:3]
monte.str.split().str.get(-1) # split 후 맨 마지막 단어
```

---

<br>

## Time Series
### In Numpy
```python
date = np.array('2001-04-17', dtype = np.datetime64) # 64비트 정수형 변환
date + np.arange(12) # 정수 연산
np.datetime64('2025-04-17T12:00:00', 'ns')
```

### In Pandas
```python
date = pd.to_datetime("2025-04-17")
date + pd.to_timedelta(np.arange(12), 'D') # Day로 12일 더하기
```

```python
index = pd.DatetimeIndex(['2014-07-14', '2014-08-14', '2015-07-14', '2015-08-14'])
data = pd.Series([0, 1, 2, 3], index = index)
data['2015'] # 연도를 인덱스로 바로 활용
```

### Pandas Time Series 구조
```python
dates = pd.to_datetime([datetime(2025, 4, 17), '17th of April, 2001', '2020-03-02', '20201205'])
# DatetimeIndex(['2025-04-17', '2001-04-17', '2020-03-02', '2020-12-05'], dtype='datetime64[ns]', freq=None)

dates.to_period('D')
# PeriodIndex(['2025-04-17', '2001-04-17', '2020-03-02', '2020-12-05'], dtype='period[D]')

dates[0]
# Timestamp('2025-04-17 00:00:00')

dates - dates[0]
# TimedeltaIndex(['0 days', '-8766 days', '-1872 days', '-1594 days'], dtype='timedelta64[ns]', freq=None)
```

### Regular Sequence
```python
pd.date_range('2025-04-17', '2025-04-30') # default : [D], 날짜 범위 생성
pd.date_range('2025-04-17', periods = 12) # 12일 생성
pd.date_range('2025-04-17', freq = 'H', periods = 12) # H : hour 기준으로 12개 생성
```

resample : group by랑 비슷하게 (aggregation)

asfreq : 지정한 한 날 가져옴 (data selection)

Time Shift : 날짜 이동

Rolling Windows (Windowing) : moving average (이동 평균)