# sectumsempra



## Kullanım

### Import

```python
from linear_regression import LinearRegression
import numpy as np 
```

### verilerin yazılması
```python
x_train = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7]
])
y_train = np.array([10, 20, 30, 40])
```

### model ve fit
```python
model = LinearRegression()
history = model.fit(x_train, y_train, alpha=1e-2, num_iters=1000, verbose=True)
```

    Epoch    0: Cost = 34.0312
    Epoch  100: Cost = 4.6980
    Epoch  200: Cost = 2.0985
    Epoch  300: Cost = 0.9373
    Epoch  400: Cost = 0.4187
    Epoch  500: Cost = 0.1870
    Epoch  600: Cost = 0.0835
    Epoch  700: Cost = 0.0373
    Epoch  800: Cost = 0.0167
    Epoch  900: Cost = 0.0074


### predict
```python
model.predict(x_train)
```
`array([10.12677403, 20.05584615, 29.98491827, 39.91399038])`

```python
y_train
```
`array([10, 20, 30, 40])`
