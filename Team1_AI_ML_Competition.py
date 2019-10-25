#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:05:06 2019

@author: juliethomson


*/ Dear Preston: 
    
https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
This is a great resource to look at for pre-modeling using python, and also resampling with SMOTE. 
I leveraged this to get the ball rollin!

-Julie 
*/ 



###Other set ups before runnning this: 
Run this in your command line >> conda install -c conda-forge imbalanced-learn
This package is needed for the set up of SMOTE


"""


# Import packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt #used for visualization
plt.rc("font", size=14) #used for visualization
from sklearn.linear_model import LogisticRegression #used for modeling
from sklearn.feature_selection import RFE #used for modeling
from sklearn.metrics import classification_report #used for classifications and output of results
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split #used to cut the data into test / train
import seaborn as sns #used to create visualizations in python
from imblearn.over_sampling import SMOTE #used for oversampling technique
sns.set(style="white") #used to create visualizations in python
sns.set(style="whitegrid", color_codes=True) #used to create visualizations in python
from sklearn.linear_model import LogisticRegression #logistic regression 
from sklearn.metrics import confusion_matrix #calls confusion matrics for logreg
from sklearn.metrics import classification_report #prints prevision, recall, f-score, etc. 
from statsmodels.api import OLS 


#set path to read in data from desktop (will be modified to change to connecting to S3 bucket)
path = r'/Users/juliethomson/Documents/AI_ML/Hackathon/Data /'
##-----------------------------------------------------------------------------
##--------1. Importing data & Exploring (Can also be done in tableau for presentation purposes)
##-----------------------------------------------------------------------------

df = pd.read_csv(path + 'raw_data_8_19_19.csv')
print(df.head(1))
print(df.shape)
print(list(df.columns))
print(df['Attrition'].unique())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis') #reviews completeness of the data


#-------delete me later on, or transform objects to a numeric value in later tranformations to keep the data 

##print(df.dtypes)
#df_objects = df.select_dtypes(exclude=['object'])
#list(df_objects.columns)
##print(df.dtypes)


##-----------------------------------------------------------------------------
##--------2. Reviewing all content, distribtrutions of the dataset and columns
##-----------------------------------------------------------------------------


sns.countplot(x='JobSatisfaction', data = df)
sns.countplot(x='PerformanceRating', data = df)

df.groupby('JobSatisfaction').mean() #shows mean of each column by job satisfaction level 
df.groupby('PerformanceRating').mean() #shows mean of each column by job satisfaction level 


df.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')
df.describe()

##-----------------------------------------------------------------------------
##--------3. dummy variables prior to modeling example
##-----------------------------------------------------------------------------
##Action item: (WILL NEED TO ASSESS WHICH VARIABLES WE NEED TO CHECK, and remove to make sure no mutlicolinearity)

cat_vars=['JobLevel','Department','MaritalStatus', 'Attrition', 'BusinessTravel', 'EducationField', 'Gender', 'JobRole',
       'Over18', 'OverTime']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    print(list(cat_list))
    df1=df.join(cat_list)
    print(df1.columns)
    df=df1
    
    print(list(df.columns))
    

data_vars=df.columns.values.tolist()
print(data_vars)
to_keep=[i for i in data_vars if i not in cat_vars]  
data_final=df[to_keep]
data_final.columns.values    
df = data_final ### @PRESTON: Remember before running this code, that you need to pick / remove the variables that we need to consider for analysis

###### removing columns with co-linearity

df = df.drop('JobLevel_1', axis = 1)
df = df.drop('Department_Sales', axis = 1)
df = df.drop('Attrition_No', axis = 1)
df = df.drop('BusinessTravel_Non-Travel', axis = 1)
df = df.drop('Gender_Male', axis = 1)
df = df.drop('OverTime_No', axis = 1)




##--------Understanding the target variable prior to cutting the data

df['PerformanceRating'].value_counts()
df.PerformanceRating[df.PerformanceRating == 3] = 0
df.PerformanceRating[df.PerformanceRating == 4] = 1
df['PerformanceRating'].value_counts() ##confirmataion of successful convert from a yes to a 1

count_other_perform = len(df[df['PerformanceRating']==0])
count_high_perform = len(df[df['PerformanceRating']==1])
pct_of_high_performers = count_high_perform/(count_other_perform +count_high_perform)
print("percentage of high performers", pct_of_high_performers*100) #15%



##--------Subsetting columns for later in the analysis (can we predict performance based on interview vs. post employment indicators?)
##------------Subsets of the data will be used later on in modeling

print(df.dtypes)

#Pre-employment Subset
df_pre_employment = df[['PerformanceRating', 'Age', 'DistanceFromHome', 'Education', 'MonthlyIncome', 'NumCompaniesWorked', 'TotalWorkingYears', 'EducationField_Human Resources','EducationField_Life Sciences'
                            , 'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Female' ]] 

data_pre_employment_list=df_pre_employment.columns.values.tolist()
print(data_pre_employment_list)
df_pre_employment.dtypes




#Post-Employment: All other columns not identifed in the above 
post_employment_list=[i for i in df if i not in data_pre_employment_list] 
print(post_employment_list)
post_employment_list = post_employment_list + ['PerformanceRating']

df_post_employment = df[post_employment_list]
df_post_employment.dtypes
print(df_post_employment.head())



##-------last check to ensure all columns are right datatype 
df.dtypes

##-----------------------------------------------------------------------------
##-------4. Cut the train / test with SMOTE
##-----------------------------------------------------------------------------



#Option 1: No re-sampling, simple cut of the data 
#because of the undistributed data, SMOTE, or another re-sampling is highly encouraged
X_train, X_test, y_train, y_test = train_test_split(df.drop('PerformanceRating',axis=1), df['PerformanceRating'], test_size=0.30,random_state=91)
print(X_train.shape) 
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

sns.countplot(x=y_train[0:], data = y_train)
sns.countplot(x=y_train[0:], data = y_test)


####Option 2: Smote resampling
#Recomend using this as a reference: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8  


X = df.loc[:, df.columns != 'PerformanceRating']
y = df.loc[:, df.columns == 'PerformanceRating']

print(X.shape)
print(y.shape)

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of other performers in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of top performers",len(os_data_y[os_data_y['y']==1]))
print("Proportion of other peformers data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of top performers in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


print(X_train.shape) 
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#####Option 3: 
##Smote resampling with only pre-interview content; this is the same SMOTE approach to the above, with only a portion of the dataset)


X2 = df_pre_employment.loc[:, df_pre_employment.columns != 'PerformanceRating']
y2 = df_pre_employment.loc[:, df_pre_employment.columns == 'PerformanceRating']

print(X2.shape)
print(y2.shape)

os = SMOTE(random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=0)
columns = X2_train.columns

os_data_X2,os_data_y2=os.fit_sample(X2_train, y2_train)
os_data_X2 = pd.DataFrame(data=os_data_X2,columns=columns )
os_data_y2= pd.DataFrame(data=os_data_y2,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X2))
print("Number of other performers in oversampled data",len(os_data_y2[os_data_y2['y']==0]))
print("Number of top performers",len(os_data_y2[os_data_y2['y']==1]))
print("Proportion of other peformers data in oversampled data is ",len(os_data_y2[os_data_y2['y']==0])/len(os_data_X2))
print("Proportion of top performers in oversampled data is ",len(os_data_y2[os_data_y2['y']==1])/len(os_data_X2))


print(X2_train.shape) 
print(X2_test.shape)
print(y2_train.shape)
print(y2_test.shape)




##-----------------------------------------------------------------------------
##--------5. Logistic Regression
##-----------------------------------------------------------------------------

###messing around with another example ---- delete me

##import statsmodels.api as sm

##logit_model = sm.Logit(y2_train, sm.add_constant(X2_train)).fit()
##logit_model.summary()

###messing around with another example ---- delete me







##part 1: entire dataset --------------------------
#Everthing set to default at first
logreg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
model_results = logreg.fit(X_train, y_train.values.ravel())
print(logreg.coef_)
print(logreg.intercept_)
print(logreg.predict_proba(X_test))



import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit(max_iter=100)





print(model_results)
y_pred = logreg.predict(X_test))


confusion_matrix_1 = confusion_matrix(y_test, y_pred)
print(confusion_matrix_1)
print(classification_report(np.array(y_test), y_pred))







##################################################JRT START HERE. 
list((y_test[0:]))
JRT = (y_test.iloc[0:]).to
print(type(JRT)
###needs to change to an array


print(np.array(y_test))
JRT_test_1 = 
##################################################JRT END HERE 
#----viz of roc curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



##part 2: pre-interview dataset --------------------

logreg = LogisticRegression()
results_pre = logreg.fit(X2_train, y2_train.values.ravel())

y2_pred = logreg.predict(X2_test)
type(y2_pred)
type(y2_test)

confusion_matrix_pre = confusion_matrix(y2_test, y2_JRT)
print(confusion_matrix)
print(classification_report(y_test, y_pred))



dataset = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1]})
>>> print(dataset)
   Column1  Column2
0      5.8      2.8
1      6.0      2.2









