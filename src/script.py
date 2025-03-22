# Importing the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib


warnings.filterwarnings('ignore')




# Importing the csv file
data = pd.read_excel('INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls')

data.shape

data.columns

data.head()

# Looking for missing data
data.info()






# A new pandas Dataframe is created to analyze department wise performance as asked.
dept = data.iloc[:,[5,27]].copy()
dept_per = dept.copy()

# Finding out the mean performance of all the departments and plotting its bar graph using seaborn.
dept_per.groupby(by='EmpDepartment')['PerformanceRating'].mean()

plt.figure(figsize=(10,4.5))
sns.barplot(x='EmpDepartment', y='PerformanceRating', data=dept_per)

# Analyze each department separately
dept_per.groupby(by='EmpDepartment')['PerformanceRating'].value_counts()

# Creating a new dataframe to analyze each department separately
department = pd.get_dummies(dept_per['EmpDepartment'])
performance = pd.DataFrame(dept_per['PerformanceRating'])
dept_rating = pd.concat([department,performance],axis=1)


# Plotting a separate bar graph for performance of each department using seaborn
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.barplot(x='PerformanceRating', y='Sales', data=dept_rating)

plt.subplot(2, 3, 2)
sns.barplot(x='PerformanceRating', y='Development', data=dept_rating)

plt.subplot(2, 3, 3)
sns.barplot(x='PerformanceRating', y='Research & Development', data=dept_rating)

plt.subplot(2, 3, 4)
sns.barplot(x='PerformanceRating', y='Human Resources', data=dept_rating)

plt.subplot(2, 3, 5)
sns.barplot(x='PerformanceRating', y='Finance', data=dept_rating)

plt.subplot(2, 3, 6)
sns.barplot(x='PerformanceRating', y='Data Science', data=dept_rating)

plt.tight_layout()
plt.show()





# Encoding all the ordinal columns and creating a dummy variable for them to see if there are any effects on Performance Rating
enc = LabelEncoder()
for i in (2,3,4,5,6,7,16,26):
    data.iloc[:,i] = enc.fit_transform(data.iloc[:,i])
data.head()

# Finding out the correlation coeffecient to find out which predictors are significant.
data.corr(numeric_only=True)

# Dropping the first columns as it is of no use for analysis.
data.drop(['EmpNumber'],inplace=True,axis=1)

data.head()

# Here we have selected only the important columns
y = data.PerformanceRating
#X = data.iloc[:,0:-1]  All predictors were selected it resulted in dropping of accuracy.
X = data.iloc[:,[4,5,9,16,20,21,22,23,24]] # Taking only variables with correlation coeffecient greater than 0.1
X.head()

# Splitting into train and test for calculating the accuracy
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

# Standardization technique is used
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train.shape

X_test.shape





# Logistic Regression

# Training the model
model_logr = LogisticRegression()
model_logr.fit(X_train,y_train)

# Predicting the model
y_predict_log = model_logr.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_log))
print(classification_report(y_test,y_predict_log))

confusion_matrix(y_test,y_predict_log)





# Support Vector Machine

# Training the model
rbf_svc = SVC(kernel='rbf', C=100, random_state=10).fit(X_train,y_train)

# Predicting the model
y_predict_svm = rbf_svc.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_svm))
print(classification_report(y_test,y_predict_svm))

confusion_matrix(y_test,y_predict_svm)






# Decision Tree with Grid Search

# Training the model

classifier_dtg=DecisionTreeClassifier(random_state=42,splitter='best')
parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini']},{'min_samples_split':[2,3,4,5],'criterion':['entropy']}]

model_griddtree=GridSearchCV(estimator=classifier_dtg, param_grid=parameters, scoring='accuracy',cv=10)
model_griddtree.fit(X_train,y_train)


model_griddtree.best_params_

# Predicting the model
y_predict_dtree = model_griddtree.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_dtree))
print(classification_report(y_test,y_predict_dtree))

confusion_matrix(y_test,y_predict_dtree)





# Random Forest with Grid Search

# Training the model

classifier_rfg=RandomForestClassifier(random_state=33,n_estimators=23)
parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}]

model_gridrf=GridSearchCV(estimator=classifier_rfg, param_grid=parameters, scoring='accuracy',cv=10)
model_gridrf.fit(X_train,y_train)


model_gridrf.best_params_

# Predicting the model
y_predict_rf = model_gridrf.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_rf))
print(classification_report(y_test,y_predict_rf))

confusion_matrix(y_test,y_predict_rf)






# Naive Bayes Bernoulli

# Training the model
model_nb = BernoulliNB()
model_nb.fit(X_train,y_train)

# Predicting the model
y_predict_nb = model_nb.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_nb))
print(classification_report(y_test,y_predict_nb))

confusion_matrix(y_test,y_predict_nb)






# K-Nearest Neighbors

# Training the model
model_knn = KNeighborsClassifier(n_neighbors=10,metric='euclidean') # Maximum accuracy for n=10
model_knn.fit(X_train,y_train)

# Predicting the model
y_predict_knn = model_knn.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_knn))
print(classification_report(y_test,y_predict_knn))

confusion_matrix(y_test,y_predict_knn)









# XGBoost Classifier

# Training the model

y_train_normalized = np.unique(y_train, return_inverse=True)[1]

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train_normalized)


# Predicting the model
y_predict_xgb = model_xgb.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_xgb))
print(classification_report(y_test,y_predict_xgb))

confusion_matrix(y_test,y_predict_xgb)






# Artificial Neural Network

# Training the model
model_mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),batch_size=10,learning_rate_init=0.01,max_iter=2000,random_state=10)
model_mlp.fit(X_train,y_train)

# Predicting the model
y_predict_mlp = model_mlp.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test,y_predict_mlp))
print(classification_report(y_test,y_predict_mlp))

confusion_matrix(y_test,y_predict_mlp)

# Exporting the trained model
joblib.dump(model_gridrf,'INX_Future_Inc.ml')
