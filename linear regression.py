import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
dataSet = pd.read_csv('https://raw.githubusercontent.com/gokuismyname/linear-regression-Website-Phishing/main/stock%20prediction.csv')
dataSet.head()
dataSet.isnull()

crtformat = dataSet.corr()
datafeature = crtformat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(dataSet[datafeature].corr(), annot=True, cmap="RdYlGn")

dataSet.plot(x='Hours', y='Scores', style='o')
plt.show()

independent = dataSet.iloc[:, :-1].values
dependent = dataSet.iloc[:, 1].values

independent_train, independent_test, dependent_train, dependent_test = train_test_split(independent, dependent,test_size=0.3, random_state=0)
regressionval = LinearRegression()
regressionval.fit(independent_train, dependent_train)

line = regressionval.coef_ * independent + regressionval.intercept_
plt.scatter(independent, dependent)
plt.plot(independent, line)
plt.show()

print(independent_test)
print()

dependent_pred = regressionval.predict(independent_test)

model = pd.DataFrame({'Actual': dependent_test, 'Predicted': dependent_pred})

print(model)
print()

hours = [[9.27]]
own_pred = regressionval.predict(hours)
print()

print("Number of hours={}".format(hours))
if own_pred[0] > 100:
    print("predicted outcome=100")
else:
    print("Prediction value = {}".format(own_pred[0]))
print()

print('Mean Absolute Error:', metrics.mean_absolute_error(dependent_test, dependent_pred))
print()

print('Variance score :%2f' % regressionval.score(independent_test, dependent_test))
print()
