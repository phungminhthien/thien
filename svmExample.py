#Import scikit-learn dataset library
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
#Load dataset
cancer = datasets.load_digits()
# cancer = datasets.load_breast_cancer()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)
# 70% training and 30% test
# print the names of the 13 features
print("Features: ", cancer.feature_names)
# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
x_feature = -2 # index âm, lấy cột thứ 2 từ phải sang
y_feature = -1 # lấy cột cuối cùng
plt.scatter(cancer.data[:, x_feature], cancer.data[:, y_feature], c=cancer.target)
plt.xlabel(cancer.feature_names[x_feature])
plt.ylabel(cancer.feature_names[y_feature])
plt.show()

# Model Accuracy: how often is the classifier correct?
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

# print data(feature)shape
# print(cancer.data.shape)

# print the cancer data features (top 5 records)
print(np.array(cancer.data[0:5],dtype='object'))

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)
# Model Precision: what percentage of positive tuples are labeled as such?
# print("Precision:",metrics.precision_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
# micro ra điểm tổng cho tất cả class
# None ra điểm cho  từng class


# Model Recall: what percentage of positive tuples are labelled as such?
# print("Recall:",metrics.recall_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

