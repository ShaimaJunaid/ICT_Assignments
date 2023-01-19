#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the datasets 

# In[2]:


train_data = pd.read_csv("C:/DSA COURSE/Loan Prediction/train_ctrUa4K.csv")
test_data  = pd.read_csv("C:/DSA COURSE/Loan Prediction/test_lAUu6dG.csv")
submission = pd.read_csv("C:/DSA COURSE/Loan Prediction/sample_submission_49d68Cx.csv")


# # Train Dataset

# Data analysis 

# In[3]:


# datas of train dataset
train_data


# In[4]:


# Data of first five 
train_data.head(5)


# In[5]:


# Information of train dataset
train_data.info()


# In[6]:


#Shape of dataset
train_data.shape


# In[7]:


# Column name of train datset
train_data.columns


# In[8]:


# datatype of train dataset
train_data.dtypes


# In[9]:


train_data["Loan_ID"].value_counts()


# In[10]:


# check the values of dependents column
train_data["Dependents"].value_counts()


# In[11]:


#replacing the value 3+ with 4
train_data = train_data.replace(to_replace = '3+',value =4 )


# In[12]:


# again check the values of dependents
train_data["Dependents"].value_counts()


# In[13]:


# Check the datatype of dependent
train_data["Dependents"].dtypes


# In[14]:


# Change the datatype of dependent from object to float
train_data["Dependents"] = train_data["Dependents"].astype(float)


# In[15]:


# datatypes of train dataset
train_data.dtypes


# In[16]:


# Describe the integer datatype of train datset 
train_data.describe()


# # Pre processing & Exploratory Data Analysis 

# In[17]:


# Null value
train_data.isnull().sum()


# In[18]:


# store the null dates to another variable 
null_values =train_data[['Gender', 'Married', 'Dependents','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']]
null_values


# In[19]:


# checking the null values
null_values.isnull().sum()


# In[20]:


# Percentage of null values
null_values.isnull().sum()*100/len(train_data)


#  All null values are below 50% So no need to drop

# In[21]:


# count plot of Credit_History
sns.countplot(x ='Credit_History',data = train_data)
plt.show()


# In[22]:


# frequency graph of integer dataset
freq_graph = train_data.select_dtypes(include="float")
freq_graph


# In[23]:


# Plot the frequency graph
freq_graph.hist(figsize=[20,15])
plt.show()


# In[24]:


# Here the distribution is skewed ..so numerical datas can be filled with median


# In[25]:


# fill the numerical null with median
for i in["Dependents", "LoanAmount",'Loan_Amount_Term','Credit_History']:
    train_data[i]=train_data[i].fillna(train_data[i].median())
# fill the word "missing " in the place of categorical null values
for i in["Gender", "Married",'Self_Employed']:
    train_data[i]=train_data[i].fillna('Missing')


# In[26]:


train_data.isnull().sum()

# Loan Status -------------> Dependent variable
# In[27]:


# count plot of Loan_Status
sns.countplot(x ='Loan_Status',data = train_data)
plt.show()


#  categorical datas 

# In[28]:


cat = train_data.select_dtypes(exclude='number')
cat.columns


# In[29]:


fig, axes = plt.subplots(5, figsize=(10, 20))
fig.suptitle('Count Plot of Numerical values')
sns.countplot(ax=axes[0], data=train_data, x='Gender', hue ="Loan_Status")
sns.countplot( ax=axes[1],data=train_data, x='Married', hue ="Loan_Status")
sns.countplot( ax=axes[2],data=train_data, x='Education', hue ="Loan_Status")
sns.countplot(ax=axes[3], data=train_data, x='Self_Employed', hue ="Loan_Status")
sns.countplot( ax=axes[4],data=train_data, x='Property_Area', hue ="Loan_Status")
plt.show()


# 1. Loan approved high in male compared to female
# 2. High loan approved in married
# 3. More loan approved in Graduate people in case of education
# 4. Loan approval high in Non-Self Employed 
# 5. Semi urban property area have high loan approval status compared to Urban and Rural

# Numerical Datas

# In[30]:


num = train_data.select_dtypes(exclude='object')
num.columns


# In[31]:


fig, axes = plt.subplots(3 ,figsize=(10, 20))
fig.suptitle('Count Plot of Numerical values')
sns.countplot(ax=axes[0], data=train_data, x='Dependents', hue ="Loan_Status")
sns.countplot(ax=axes[1], data=train_data, x='Loan_Amount_Term', hue ="Loan_Status")
sns.countplot(ax=axes[2], data=train_data, x='Credit_History', hue ="Loan_Status")
plt.show()


# In[32]:


fig, axes = plt.subplots(3 ,figsize=(10, 20))
fig.suptitle('Box Plot of Numerical values')
sns.boxplot(ax=axes[0],data=train_data, y='ApplicantIncome', x="Loan_Status")
sns.boxplot(ax=axes[1],data=train_data, y='CoapplicantIncome', x="Loan_Status")
sns.boxplot(ax=axes[2],data=train_data, y='LoanAmount', x="Loan_Status")


# In[33]:


train_data.head(4)


# In[34]:


train_data["Total_Income"] =train_data["ApplicantIncome"] + train_data["CoapplicantIncome"]
train_data


# In[35]:


train_data["Credit_History"].value_counts()


# In[36]:


train_data = train_data.drop(["Loan_ID","ApplicantIncome","CoapplicantIncome"],axis=1)
train_data


# In[37]:


#Label Encoding
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
train_data["Gender"] = label.fit_transform(train_data["Gender"])
train_data["Married"] = label.fit_transform(train_data["Married"])
train_data["Education"] = label.fit_transform(train_data["Education"])
train_data["Self_Employed"] = label.fit_transform(train_data["Self_Employed"])
train_data["Property_Area"] = label.fit_transform(train_data["Property_Area"])
train_data["Loan_Status"] = label.fit_transform(train_data["Loan_Status"])
train_data


# In[38]:


train_data.shape


# In[39]:


#check the correlation
corr_matrix =train_data.corr()
corr_matrix


# In[40]:


#Heatmap
plt.subplots(figsize=(20,10))
sns.heatmap(corr_matrix,annot = True,cmap="YlGnBu")
plt.show()


# # Test Dataset

# Data analysis 

# In[41]:


# datas of test dataset
test_data


# In[42]:


# Data of first five 
test_data.head(5)


# In[43]:


#Shape of dataset
test_data.shape


# In[44]:


# Column name of test datset
test_data.columns


# In[45]:


# datatype of test dataset
test_data.dtypes


# In[46]:


# check the values of dependents column
test_data["Dependents"].value_counts()


# In[47]:


#replacing the value 3+ with 4
test_data = test_data.replace(to_replace = '3+',value =4 )


# In[48]:


# check the values of dependents column
test_data["Dependents"].value_counts()


# In[49]:


# Change the datatype of dependent from object to float
test_data["Dependents"] = test_data["Dependents"].astype(float)


# In[50]:


# datatypes of test dataset
test_data.dtypes


# # Pre processing & Exploratory Data Analysis 

# In[51]:


test_data.isnull().sum()


# In[52]:


# fill the numerical null with median
for i in["Dependents", "LoanAmount",'Loan_Amount_Term','Credit_History']:
    test_data[i]=test_data[i].fillna(test_data[i].median())
# fill the word "missing " in the place of categorical null values
for i in["Gender",'Self_Employed']:
    test_data[i]=test_data[i].fillna('Missing')


# In[53]:


test_data.isnull().sum()


# In[54]:


test_data["Total_Income"] =test_data["ApplicantIncome"] + test_data["CoapplicantIncome"]
test_data


# In[55]:


test_data = test_data.drop(["Loan_ID","ApplicantIncome","CoapplicantIncome"],axis=1)
test_data


# In[56]:


#Label Encoding
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
test_data["Gender"] = label.fit_transform(test_data["Gender"])
test_data["Married"] = label.fit_transform(test_data["Married"])
test_data["Education"] = label.fit_transform(test_data["Education"])
test_data["Self_Employed"] = label.fit_transform(test_data["Self_Employed"])
test_data["Property_Area"] = label.fit_transform(test_data["Property_Area"])
test_data


# In[57]:


test_data.shape


# # Model Building

# #Target varibale-------->Loan Status

# In[58]:


# Independent variables
x = train_data.drop(["Loan_Status"],axis = 1)
x


# In[59]:


#Dependent / Target variable
y = train_data["Loan_Status"]
y


# In[60]:


#Scaling
from sklearn.preprocessing import StandardScaler
Std_Scaler = StandardScaler()
x = Std_Scaler.fit_transform(x)


# In[61]:


type(x)


# In[62]:


#convert array to dataframe
x = pd.DataFrame(x)


# In[63]:


#column name
x.columns = ["Gender",'Married', 'Dependents', 'Education',"Self_Employed","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area","Total_Income"]


# In[64]:


# split the datas
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42, test_size = 0.2)


# # Logistic Regression

# In[65]:


#import Logistic Regression 
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
# Train the model using trainnig sets
log_model = log_model.fit(x_train,y_train)
y_pred = log_model.predict(x_test)


# In[66]:


# Import confusion_matrix,accuracy_score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#predic metric to get performance
print("Accuracy is " ,accuracy_score(y_test,y_pred)*100 ,"%")


# In[67]:


print("--------------Classification Report---------\n" ,classification_report(y_test,y_pred))


# In[68]:


#confusion matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix  \n",confusion_matrix)


# In[69]:


#Prediction using test_data
pred_test_data = log_model.predict(test_data)


# In[70]:


pred_test_data


# In[71]:


pred_test_data .shape


# In[72]:


submission


# In[73]:


submission.shape


# In[74]:


#Replace the ‘Loan_Status’ column with the prediction values
submission['Loan_Status']=pred_test_data
submission


# In[75]:


submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)


# In[76]:


submission


# In[77]:


submission.dtypes


# In[78]:


submission.shape


# In[79]:


submission.to_csv("C:/DSA COURSE/Loan Prediction/Logistic_Model.csv",index=None)


# In[80]:


submission.shape


# In[81]:


submission.head(4)


# # K Nearest Neighbour

# In[82]:


# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[83]:


# assigning empty list
metric_k =[]
# kepping some values to neighbours 
neighbors = np.arange(3,15)
neighbors


# In[84]:


# create a classifier using euclidean
for k in neighbors:
    classifier = KNeighborsClassifier(n_neighbors = k,metric ="euclidean")
    classifier.fit(x_train,y_train)
    y_prediction = classifier.predict(x_test)
    acc = accuracy_score(y_test,y_prediction)
    metric_k.append(acc)


# In[85]:


metric_k


# In[86]:


# plot the k value
plt.plot(neighbors,metric_k,'o-')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.grid()


# In[87]:


#K = 13 has the maximum value
#put k = 13 in n_neighbors
classifier = KNeighborsClassifier(n_neighbors =13,metric ="euclidean")
classifier.fit(x_train,y_train)
y_prediction = classifier.predict(x_test)


# In[88]:


#Accuracy
print("Accuracy is " ,accuracy_score(y_test,y_prediction)*100 ,"%")


# In[89]:


print("--------------Classification Report---------\n" ,classification_report(y_test,y_prediction))


# In[90]:


#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix_knn = confusion_matrix(y_test,y_prediction)
print("Confusion Matrix  \n",confusion_matrix_knn)


# In[91]:


#Prediction using test_data
pred_KNN_test_data = classifier.predict(test_data)


# In[92]:


pred_KNN_test_data


# In[93]:


#Replace the ‘Loan_Status’ column with the prediction values
submission['Loan_Status']=pred_KNN_test_data
submission


# In[94]:


#Converting 1 to Y and O to N 
submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)


# In[95]:


# Save datset into csv
submission.to_csv("C:/DSA COURSE/Loan Prediction/KNN_Model.csv",index=None)


# In[96]:


submission.head(4)


# # SVM

# In[97]:


#import 
from sklearn.svm import SVC


# In[98]:


# svm using linear kernel
svm_cls = SVC(kernel = "linear")
svm_cls = svm_cls.fit(x_train,y_train)
y_pred_svm = svm_cls.predict(x_test)


# In[99]:


#Accuracy
print("Accuracy is " ,accuracy_score(y_test,y_pred_svm)*100 ,"%")


# In[100]:


print("--------------Classification Report---------\n" ,classification_report(y_test,y_pred_svm))


# In[101]:


# Confusion matrix
confusion_matrix_svm = confusion_matrix(y_test,y_pred_svm)
print("Confusion Matrix = \n",confusion_matrix_svm)


# In[102]:


#Prediction using test_data
pred_SVM_test_data = svm_cls.predict(test_data)


# In[103]:


pred_SVM_test_data


# In[104]:


#Replace the ‘Loan_Status’ column with the prediction values
submission['Loan_Status']=pred_SVM_test_data
submission


# In[105]:


#Converting 1 to Y and O to N 
submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)


# In[106]:


# Save datset into csv
submission.to_csv("C:/DSA COURSE/Loan Prediction/SVM_Model.csv",index=None)


# In[107]:


submission.head(4)


# # Decision Tree

# In[108]:


# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt_cls = DecisionTreeClassifier()
dt_cls = dt_cls.fit(x_train,y_train)
y_pred_dt = dt_cls.predict(x_test)


# In[109]:


#Accuracy
print("Accuracy is " ,accuracy_score(y_test,y_pred_dt)*100 ,"%")


# In[110]:


print("--------------Classification Report---------\n" ,classification_report(y_test,y_pred_dt))


# In[111]:


#confusion matrix
confusion_df = confusion_matrix(y_test,y_pred_dt)
print("Confusion Matrix = \n",confusion_df)


# In[112]:


#Prediction using test_data
pred_DT_test_data = dt_cls.predict(test_data)


# In[113]:


#Replace the ‘Loan_Status’ column with the prediction values
submission['Loan_Status'] = pred_DT_test_data
submission


# In[114]:


#Converting 1 to Y and O to N 
submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)


# In[115]:


# Save datset into csv
submission.to_csv("C:/DSA COURSE/Loan Prediction/DT_Model.csv",index=None)


# # Random Forest

# In[116]:


# import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_cls = RandomForestClassifier()
rf_cls = rf_cls.fit(x_train,y_train)
y_pred_rf = rf_cls.predict(x_test)


# In[117]:


#Accuracy
print("Accuracy is " ,accuracy_score(y_test,y_pred_rf)*100 ,"%")


# In[118]:


print("--------------Classification Report---------\n" ,classification_report(y_test,y_pred_rf))


# In[119]:


#Confusion Matrix
confusion_rf = confusion_matrix(y_test,y_pred_rf)
print("Confusion Matrix = \n",confusion_rf)


# In[120]:


#Prediction using test_data
pred_RF_test_data = rf_cls.predict(test_data)


# In[121]:


#Replace the ‘Loan_Status’ column with the prediction values
submission['Loan_Status'] = pred_RF_test_data
submission


# In[122]:


#Converting 1 to Y and O to N 
submission.Loan_Status.replace(1, 'Y', inplace=True)
submission.Loan_Status.replace(0, 'N', inplace=True)


# In[123]:


# Save datset into csv
submission.to_csv("C:/DSA COURSE/Loan Prediction/RF_Model.csv",index=None)

