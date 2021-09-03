# Data-analytics
Data analytics for the Diabetes dataset.


#CODE.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



dataset = pd.read_csv('/Users/abhishekpancholi/Desktop/Stuf/diabetes.csv')
print(dataset.shape)
print(dataset.describe())
dataset.plot(x='Glucose', y ='Insulin', style='o')
plt.title('Relation Between Glucose and Insulin')
plt.xlabel('Glucose')
plt.ylabel('Insulin')
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.displot(dataset['Glucose'])
plt.show()

X= dataset['Glucose'].value.reshape(-1,1)
Y= dataset['Insulin'].value.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print('intercept: ', regressor.intercept_)
print('coefficent: ', regressor.coef_)

y_pred = regressor.predict(X_test)

df= pd.DataFrame({'Actual': Y_test.flatten(), 'Predict': y_pred.flatten})
print(df)


#In Progress



