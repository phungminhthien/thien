#!/usr/bin/env python
# coding: utf-8

# In[93]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neighbors, datasets

# In[94]:
iris =datasets.load_iris()
# iris = datasets.load_wine()
iris_X = iris.data
iris_y = iris.target
# print(iris_X)
# print(iris_y)
print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))
X0 = iris_X[iris_y == 0,:]
print('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print('\nSamples from class 2:\n', X2[:5,:])

# In[96]:

x_feature = 0
y_feature = 1
x_column = iris.data[:, x_feature] # lấy tất cả các dòng của cột x_feature(0)
y_column = iris.data[:, y_feature]
plt.scatter(x_column, y_column, c=iris.target)
plt.xlabel(iris.feature_names[x_feature])
plt.ylabel(iris.feature_names[y_feature])

# In[97]:

x_feature = -2 # index âm, lấy cột thứ 2 từ phải sang
y_feature = -1 # lấy cột cuối cùng
plt.scatter(iris.data[:, x_feature], iris.data[:, y_feature], c=iris.target)
plt.xlabel(iris.feature_names[x_feature])
plt.ylabel(iris.feature_names[y_feature])


# In[98]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)

print('Training size: %d' %len(y_train))
print('Test size    : %d' %len(y_test))


# In[99]:


def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)
knn = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Print results for 20 test data points:')
print('Predicted labels: ', y_pred[20:40])
print('Ground truth    : ', y_test[20:40])
from sklearn.metrics import accuracy_score
print('Accuracy of 10NN (1/distance weights): %.2f %%' %(100*accuracy_score(y_test, y_pred)))


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[100]:
# Test
print('\nKiểm tra phân loại hoa thử: ')
print('\nHoa có 4 kích thước là: [6.3, 2.3, 4.4, 1.3]')
# test1 = knn.predict([[14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065]])
test1 = knn.predict([[6.3, 2.3, 4.4, 1.3]])
print('Kết quả dự đoán là loại: ',iris.target_names[test1])
print('Kết quả thực tế là: versicolor')

# print('\nHoa có 4 kích thước là: [5.4,3.4,1.7,0.2]')
# test2 = knn.predict([[5.4,3.4,1.7,0.2]])
# print('Kết quả dự đoán là loại: ',iris.target_names[test2])
# print('Kết quả thực tế là: setosa')
#
# print('\nHoa có 4 kích thước là: [6.5,3.0,5.8,2.2]')
# test3 = knn.predict([[6.5,3.0,5.8,2.2]])
# print('Kết quả dự đoán là loại: ',iris.target_names[test3])
# print('Kết quả thực tế là: virginica')


# In[101]:
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')




