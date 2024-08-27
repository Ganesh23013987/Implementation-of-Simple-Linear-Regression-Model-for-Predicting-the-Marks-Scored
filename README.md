# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GANESH D 
RegisterNumber:  212223240035
*/
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv("/student_scores.csv")
print(dataset.head())
print(dataset.tail())
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/eb3c8549-67d1-4783-a026-d62e43a17ce4">

```
dataset.info()
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/1885c278-c3ef-4285-846f-733d6d1027ca">

```
dataset.describe()
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/adf7f9c2-6e23-4b7e-b31b-4bf125e9a28c">

```
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/6feb64a3-6b3a-4b74-9dae-f4b1730030cd">

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/96eea1e4-6a90-4a67-9898-fdbe2fbb57c7">

```
x_test.shape
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/03386f60-328d-462c-8740-bb97daec13df">

```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/4f5ffe5c-2708-44ab-b854-8ff0ceeea557">

```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/01309e63-4334-4c2e-9a7a-6e1f318ea19d">

```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/d5db02bb-67e3-433d-ba01-08134d899eb9">

```
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="silver")
plt.title('Test_set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

<img width="780" alt="image" src="https://github.com/user-attachments/assets/6e115841-b62b-46a4-ac3d-5abc369cf24b">

<img width="780" alt="image" src="https://github.com/user-attachments/assets/9d5aa345-9c4a-451b-a0dc-68ee0f649402">


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
