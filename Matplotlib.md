# Matplotlib

<br>

## Simple Line Plots
```python
fig = plt.figure() # window, canvas
ax = plt.axes() # 축, 영역
```

```python
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)

ax.plot(x, np.sin(x))
plt.plot(x, np.sin(x)) # 동일하게 sin 그래프 표시
```

```python
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
# 한 캔버스 내에 두 개의 그래프 그림
```

### Color & Style
```python
color = ['blue', 'g', '0.75', '#FFDD44']
linestyle = ['solid, -', 'dashed, --', 'dashdot, -.', 'dotted, :']

plt.plot(x, y, color = "", linestyle = "")

plt.plot(x, x - 2, '-g')  # solid green
plt.plot(x, x - 3, '--c') # dashed cyan
plt.plot(x, x - 4, '-.k') # dashdot black
plt.plot(x, x - 5, ':r')  # dotted red
```

### limit
```python
plt.plot(x, np.sin(x))

plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5) # x, y 범위 제한

plt.axis('tight') # 최대 x, y에 맞게

plt.axis('equal') # x축과 y축의 스케일(비율)을 동일하게
```

### Labeling
```python
plt.title("A Sine Curve") # 제목 표시
plt.xlabel("x") # x축의 정보
plt.ylabel("sin(x)") # y축의 정보

plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.legend() # 범례 표시 (label 정보를 기반)
```

### axes()를 이용한 방법
```python
ax = plt.axes()
ax.plot(x, np.sin(x))

# 객체지향에서 유용하게 사용
ax.set(xlim=(0, 10), ylim=(-2, 2),
       xlabel='x', ylabel='sin(x)',
       title='A Simple Plot')
```

---

<br>

## Scatter
```python
markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd', 'p', 'h']

plt.plot(x, y, 'o', color='black') # 'o'가 scatter 의미 / 점들을 marker라 함
plt.plot(x, y, '-ok') # 라인 그래프 -와 스캐터 o도 함께 가능 / 색상 blac'k'
plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
```

```python
plt.scatter(x, y, marker='o')
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=iris.target, cmap='viridis')
```

---

<br>

## Errorbars
```python
x = np.linspace(0, 10, 50)
y = np.sin(x) + dy * np.random.randn(50)
dy = 0.5

plt.errorbar(x, y, yerr=dy, fmt='.k') # y-error : 오차 범위
plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
```
>xerr도 가능하다.

---

<br>

## Density
```python
plt.contour(X, Y, Z, colors='black') # color or cmap 하나 사용
plt.colorbar() # 그래프 우측에 컬러바가 나온다.
```

---

<br>

## Histogram
```python
data = np.random.randn(1000)

plt.hist(data)
plt.hist(data, bins=30, alpha=0.5, # bins : 구간, alpha : 투명도
         histtype='stepfilled', color='steelblue', # 계단형, 스틸블루색상
         edgecolor='none') # 테두리선
```

```python
# 정규분포를 따르는 3개의 데이터 샘플 생성
x1 = np.random.normal(0, 0.8, 1000)   # 평균 0, 표준편차 0.8
x2 = np.random.normal(-2, 1, 1000)    # 평균 -2, 표준편차 1
x3 = np.random.normal(3, 2, 1000)     # 평균 3, 표준편차 2

# 히스토그램 공통 스타일 정의
kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)

# 세 히스토그램을 겹쳐서 그림
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)
```

```python
# np.histogram()으로 data의 정보를 추출
counts, bin_edges = np.histogram(data, bins=5)
```
counts : 각 구간에 속하는 데이터 개수 (len = bins) <br>
bin_edges : 구간의 경계값 (len = bins+1)

### 2차원 Hist
```python
plt.hist2d(x, y, bins=30, cmap='Blues')

plt.hexbin(x, y, gridsize=30, cmap='Blues') # 6각형 bin
```

---

<br>

## Plot Legends 꾸미기

```python
ax.legend()
ax.legend(loc='upper left', frameon=False)
ax.legend(frameon=False, loc='lower center', ncol=2)
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
```

### 원소 선택
```python
lines = plt.plot(x, y)
plt.legend(lines[:], ['first', 'second']) # lines 주에 골라서 ['first, second] 범례 제공
```

```python
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
```

---

<br>

## Colorbars
```python
cmap = ['gray', 'jet', 'viridis', 'cubehelix', 'RdBu']
plt.imshow(I, cmap='gray') # 흑백 프린터 사용

plt.colorbar()
plt.colorbar(extend='both') # 컬러바 양 끝을 뾰족하게

plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
```

---

<br>

## Subplot
```python
ax1 = plt.axes()  # standard axes
ax2 = plt.axes([0.25, 0.25, 0.5, 0.5]) # [x, y, w, h] 0 ~ 1기준
```
x, y 지점에서 시작하여 w%, h% 만큼의 길이

### Grid 내의 Subplot
```python
for i in range(1, 7):
    plt.subplot(2, 3, i) # 2행 3열 크기의 i번째 subplot 지정
    plt.text(0.5, 0.5, str((2, 3, i)), # plt.text() 명령을 통해 입력
             fontsize=18, ha='center')
```

```python
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4) # fig 객체
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
           fontsize=18, ha='center') # ax 객체 생성 및 text() 메소드 활용 -> 객체지향
```

### 빈 Grid
```python
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
```

### axes는 2차원 배열로 나타낼 수 있다.
```python
# axes are in a two-dimensional array, indexed by [row, col]
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)),
                      fontsize=18, ha='center')
```

### plt.GridSpec
```python
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

# grid는 현재 2x3 크기. 
plt.subplot(grid[0, 0]) # (0, 0) 크기
plt.subplot(grid[0, 1:]) # (0, 1) ~ (0, 2) 크기
plt.subplot(grid[1, :2]) # (1, 0) ~ (1, 1) 크기
plt.subplot(grid[1, 2]) # (1, 2) 크기
```

---

<br>

## Text Annotation
```python
# plt.text / ax.text
style = dict(size=10, color='gray')

ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)
```

```python
# transform=ax.transData is the default, but we'll specify it anyway
ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData) # 실제 좌표 (1, 5)
ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes) # (0, 0) ~ (1, 1) 가정하여 좌표 변환 - normalized
ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure) # 그림 기준 (0, 0) ~ (1, 1) 가정하여 좌표 변환 - normalized
```

### Arrow Annotation
```python
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"))
```

---

<br>

## Style Sheets
```python
# use a gray background
ax = plt.axes()
ax.set_axisbelow(True)

# draw solid white grid lines
plt.grid(color='w', linestyle='solid')

# hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)
    
# hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# lighten ticks and labels
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
    
# control face and edge color of histogram
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666')
```

### 다양한 스타일
```python
plt.style.context('fivethirtyeight')
```

---

<br>

## VS Seaborn
```python
sns.kdeplot(data[col], shade=True)
sns.jointplot(x="x", y="y", data=data, kind='kde')

g = sns.catplot(x="day", y="total_bill", hue="sex", data=tips, kind="box")
g.set_axis_labels("Day", "Total Bill")
```