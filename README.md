# EX-04: SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
Load and Prepare the Dataset: Fetch the dataset and create a DataFrame, adding the target variable for housing prices.<br>

Data Splitting: Separate the features (X) and target (Y) variables, and split the data into training and testing sets.<br>

Feature Scaling: Apply feature scaling (StandardScaler) to normalize the features and target variables for better performance.<br>

Model Initialization: Initialize the SGDRegressor model, and use the MultiOutputRegressor to handle multiple target variables.<br>

Model Training: Fit the model on the training data (scaled features and target values).<br>

Prediction and Evaluation: Use the trained model to predict on the test set, inverse transform the predictions, and calculate the Mean Squared Error for model evaluation.

## Program:
```python
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Ashwin Akash M
RegisterNumber:  212223230024

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
print(data)
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
df.head()
df.tail()
df.info()
x=df.drop(columns=['AveOccup','target'])
y=df['target']
x.shape
y=df[['AveOccup','target']]
y.info()
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.fit_transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
*/
```

## Output:
![image](https://github.com/user-attachments/assets/18a36108-a9df-432d-af99-144410b6ef0b)<br>
![image](https://github.com/user-attachments/assets/67093d7e-84fb-42e8-bec1-f9cc0ace7f0b)<br>
![image](https://github.com/user-attachments/assets/0d36b871-4ec6-4de4-a082-b0f8693b66b3)<br>
![image](https://github.com/user-attachments/assets/b48fe258-d6b3-45fc-b2ce-c886e302fab6)<br>
![image](https://github.com/user-attachments/assets/fed72068-73bb-433d-ac7f-30ab0a04e56a)<br>
![image](https://github.com/user-attachments/assets/ee316ac5-ba9c-4c35-909d-cd244bb3e930)<br>
![image](https://github.com/user-attachments/assets/f37c30e4-d511-48be-b1ad-6f4100477769)<br>
![image](https://github.com/user-attachments/assets/e3541017-6698-4f27-93f8-39ca2cf4cf04)<br>
![image](https://github.com/user-attachments/assets/7a37366e-0a53-42f3-b9c1-cee84b581bdf)<br>
![image](https://github.com/user-attachments/assets/224bce36-51f5-4918-8436-c0c21604264c)<br>
![image](https://github.com/user-attachments/assets/33265c76-cb3a-448f-a84a-94730b7e79bd)<br>
![image](https://github.com/user-attachments/assets/f4ac26c9-4d74-4663-88f3-7dd400c6347d)

## Result:

Hence,Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
