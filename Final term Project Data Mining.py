#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


df = pd.read_csv("diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


#to check null values
df.isnull().sum()


# In[6]:


#Replace 0 values with nan because some of the columns have values as 0, which are basically missing.
df_copy = df.copy(deep = True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[7]:


#Impute missing values with mean and median values
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace = True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace = True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace = True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace = True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace = True)


# In[8]:


p = df_copy.hist(figsize = (20,20)) #after removing zero values


# In[9]:


#Splitting data into train data (X) and train labels (y).
X = df_copy.iloc[:, :-1].values
y = df_copy.iloc[:, -1].values


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y)


# In[11]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[12]:


from sklearn.metrics import make_scorer

def tn(Y_test,y_pred): 
    return confusion_matrix(Y_test,y_pred)[0,0]
def fp(Y_test,y_pred): 
    return confusion_matrix(Y_test,y_pred)[0,1]
def fn(Y_test,y_pred): 
    return confusion_matrix(Y_test,y_pred)[1,0]
def tp(Y_test,y_pred): 
    return confusion_matrix(Y_test,y_pred)[1,1]

def tpr(Y_test, y_pred):
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round((tp / (tp + fn)), 2)

def spc(Y_test, y_pred):
    tn = confusion_matrix(Y_test, y_pred)[0, 0]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    return round((tn / (tn + fp)), 2)

def recall(Y_test, y_pred):
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round((tp / (tp + fn)), 2)

def ppv(Y_test, y_pred):
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    return round((tp / (tp + fp)), 2)

def npv(Y_test, y_pred):
    tn = confusion_matrix(Y_test, y_pred)[0, 0]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round((tn / (tn + fn)), 2)

def fpr(Y_test, y_pred):
    tn = confusion_matrix(Y_test, y_pred)[0, 0]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    return round((fp / (tn + fp)), 2)

def fdr(Y_test, y_pred):
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    return round((fp / (tp + fp)), 2)

def fnr(Y_test, y_pred):
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round((fn / (tp + fn)), 2)

def accuracy(Y_test, y_pred):
    tn = confusion_matrix(Y_test, y_pred)[0, 0]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round(((tp + tn) / (tp + fp + fn + tn)), 2)

def F1Score(Y_test, y_pred):
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round(((2 * tp) / ((2 * tp) + fp + fn)), 2)

def error(Y_test, y_pred):
    tn = confusion_matrix(Y_test, y_pred)[0, 0]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round(((fp + fn) / (tp + fp + fn + tn)), 2)

def BACC(Y_test, y_pred):
    tn = confusion_matrix(Y_test, y_pred)[0, 0]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round(0.5 * ((tp / (tp + fn)) + (tn / (fp + tn))), 2)


def TSS(Y_test, y_pred):
    tn = confusion_matrix(Y_test, y_pred)[0, 0]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round((tp / (tp + fn)) - (fp / (fp + tn)), 2)


def HSS(Y_test, y_pred):
    tn = confusion_matrix(Y_test, y_pred)[0, 0]
    fp = confusion_matrix(Y_test, y_pred)[0, 1]
    tp = confusion_matrix(Y_test, y_pred)[1, 1]
    fn = confusion_matrix(Y_test, y_pred)[1, 0]
    return round((2 * ((tp * tn) - (fp * fn))) / 
                 (((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))), 2)


# In[13]:


scoring = {'tp': make_scorer(tp),'tn': make_scorer(tn),'fp': make_scorer(fp),'fn': make_scorer(fn),'sensitivity': make_scorer(tpr),
           'specificity':make_scorer(spc),'recall':make_scorer(recall),'precision':make_scorer(ppv), 'Negative Predictive Value':make_scorer(npv),
           'False Positive Rate':make_scorer(fpr),'False Discovery Rate':make_scorer(fdr),'False Negative Rate':make_scorer(fnr),
           'Accuracy':make_scorer(accuracy),'F1 Score':make_scorer(F1Score),'Error':make_scorer(error),
           'BACC':make_scorer(BACC),'TSS':make_scorer(TSS),'HSS':make_scorer(HSS)}


# Random Forest Classifier

# In[14]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)


# In[15]:


y_pred_rf = classifier_rf.predict(X_test)
#print(np.concatenate((y_pred_rf.reshape(len(y_pred_rf),1), y_test.reshape(len(y_test),1)),1))


# In[16]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)
accuracy_score(y_test, y_pred_rf)


# In[17]:


from sklearn.model_selection import KFold, cross_validate
cv_rf = KFold(n_splits=10,random_state=1,shuffle=False)
metrics_rf = cross_validate(classifier_rf,X_train,y_train,scoring = scoring,cv=cv_rf)


# In[18]:


parameters_rf = [value for value in metrics_rf.values()]
parameters_rf = parameters_rf[2:]

for i in range(len(parameters_rf)):
    average = round(sum(parameters_rf[i])/len(parameters_rf[i]),2)
    temp = list(parameters_rf[i])
    temp.append(average)
    parameters_rf[i]=temp
#print(parameters)  

parameters_rf = np.array(parameters_rf)
#print(parameters)
column = ['1','2','3','4','5','6','7','8','9','10','Average']
row = ['TP','TN','FP','FN','Sensitivity','Specificity','Recall','Precision','NPV','FPR','FDR','FNR','Accuracy','F1 Score','Error','BACC','TSS','HSS']
pd.DataFrame(parameters_rf, row, column)


# In[19]:


from sklearn.model_selection import cross_val_score
#train model with cv of 10
cv_scores_rf = cross_val_score(classifier_rf, X_train, y_train, cv=10,scoring='accuracy')

#print each cv score (accuracy) and average them
print(cv_scores_rf)
print('cv_scores_rf mean percent:{}'.format(np.mean(cv_scores_rf*100)))


# In[ ]:





# K-Nearest Neighbor Classifier

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[21]:


y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[23]:


from sklearn.model_selection import KFold, cross_validate, cross_val_predict
cv = KFold(n_splits=10,random_state=1,shuffle=False)
metrics = cross_validate(classifier,X_train,y_train,scoring = scoring,cv=cv)


# In[24]:


import warnings
warnings.filterwarnings("ignore")

parameters = [value for value in metrics.values()]
parameters = parameters[2:]

for i in range(len(parameters)):
    average = round(sum(parameters[i])/len(parameters[i]),2)
    temp = list(parameters[i])
    temp.append(average)
    parameters[i]=temp
#print(parameters)  

parameters = np.array(parameters)
#print(parameters)
pd.DataFrame(parameters, row, column)
    


# In[25]:


from sklearn.model_selection import cross_val_score
#train model with cv of 10
cv_scores_knn = cross_val_score(classifier, X_train, y_train, cv=10,scoring='accuracy')

#print each cv score (accuracy) and average them
print(cv_scores_knn)
print('cv_scores_knn mean percent:{}'.format(np.mean(cv_scores_knn*100)))


# SVM Classifier

# In[26]:


from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'rbf', random_state = 0)
classifier_svm.fit(X_train, y_train)


# In[27]:


y_pred_svm = classifier_svm.predict(X_test)
#print(np.concatenate((y_pred_svm.reshape(len(y_pred_svm),1), y_test.reshape(len(y_test),1)),1))


# In[28]:


from sklearn.model_selection import KFold, cross_validate
cv_svm = KFold(n_splits=10,random_state=1,shuffle=False)
metrics_svm = cross_validate(classifier_svm,X_train,y_train,scoring = scoring,cv=cv_svm)


# In[29]:


#print(metrics_svm)


# In[30]:


parameters_svm = [value for value in metrics_svm.values()]
parameters_svm = parameters_svm[2:]

for i in range(len(parameters_svm)):
    average = round(sum(parameters_svm[i])/len(parameters_svm[i]),2)
    temp = list(parameters_svm[i])
    temp.append(average)
    parameters_svm[i]=temp
#print(parameters)  

parameters_svm = np.array(parameters_svm)
#print(parameters)
pd.DataFrame(parameters_svm, row, column)


# In[31]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)
accuracy_score(y_test, y_pred_svm)


# In[32]:


cv_scores_svm = cross_val_score(classifier_svm, X_train, y_train, cv=10,scoring='accuracy')

#print each cv score (accuracy) and average them
print(cv_scores_svm)
print('cv_scores_svm mean percent:{}'.format(np.mean(cv_scores_svm*100)))


# In[33]:


models_scores_table = pd.DataFrame({'Random Forest':[metrics_rf['test_Accuracy'].mean(),
                                                     metrics_rf['test_precision'].mean(),
                                                     metrics_rf['test_recall'].mean(),
                                                     metrics_rf['test_F1 Score'].mean()],
                                    'KNN':[metrics['test_Accuracy'].mean(),
                                           metrics['test_precision'].mean(),
                                           metrics['test_recall'].mean(),
                                           metrics['test_F1 Score'].mean()],
                                   'SVM':[metrics_svm['test_Accuracy'].mean(),
                                          metrics_svm['test_precision'].mean(),
                                          metrics_svm['test_recall'].mean(),
                                          metrics_svm['test_F1 Score'].mean()]},
                                   index=['Accuracy', 'Precision', 'Recall', 'F1 Score']) 

# Add 'Best Score' column
models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
# Return models performance metrics scores data frame
print(models_scores_table)


# It appears that Random Forest Classifier has the best accuracy, recall and F1 Score and can be considered the best classifier out of the three.
