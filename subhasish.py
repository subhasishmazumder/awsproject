# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:59:26 2019

@author: KIIT
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

dataset = pd.read_csv('Data1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#Seaborn Heatmap
sns.heatmap(dataset.corr())
#Linear Regression
sns.pairplot(dataset)
X = dataset[['State','Age','Purchased']]
y = dataset['Salary']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#Logistic
from sklearn.model_selection import train_test_split
X = dataset[['State','Age','Purchased']]
y = dataset['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
#K Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
centers = kmeans.cluster_centers_
centers
kmeans.labels_

f, (ax1, ax2) = plt.subplots(nrows=1, 
                             ncols=2, 
                             sharey=True,
                             figsize=(10,6))
  
ax1.set_title('K Means (K = 4)')
ax1.scatter(data[0][:,0],
            data[0][:,1],
            c=kmeans.labels_,
            cmap='rainbow')
 
ax2.set_title("Original")
ax2.scatter(data[0][:,0],
            data[0][:,1],
            c=data[1],
            cmap='rainbow')

ax1.scatter(x=centers[:, 0], 
            y=centers[:, 1],
            c='black', 
            s=100, 
            alpha=0.5);
            
sum_square = {}

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k).fit(data[0])
    sum_square[k] = kmeans.inertia_ 
    
    sum_square
    
    plt.plot(list(sum_square.keys()),
         list(sum_square.values()),
         
         linestyle='-', 
         marker='H', 
         color='g', 
         markersize = 8, 
         markerfacecolor='b')