import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numbers as np

#loading the dataset
test_set = pd.read_csv("/test.csv")
train_set = pd.read_csv("train.csv")
train_set.head()

#Data preprocessing
train_set .isnull().sum()
#categorical null values are imputed using mode imputation
train_set['Gender'].fillna(train_set['Gender'].mode()[0], inplace=True)
train_set['Married'].fillna(train_set['Married'].mode()[0], inplace=True)
train_set['Dependents'].fillna(train_set['Dependents'].mode()[0], inplace=True)
train_set['Loan_Amount_Term'].fillna(train_set['Loan_Amount_Term'].mode()[0], inplace=True)
train_set['Credit_History'].fillna(train_set['Credit_History'].mode()[0], inplace=True)
train_set['Self_Employed'].fillna(train_set['Self_Employed'].mode()[0], inplace=True)
#changing loan status into binary
train_set['Loan_Status']=train_set['Loan_Status'].map({'Y':1,'N':0})
#Contineous null values are imputed using mean imputation
train_set['LoanAmount'].fillna(train_set['LoanAmount'].mean(), inplace=True)

#remove the outliers using log tranformation
train_set['LoanAmount_log']=np.log(train_set['LoanAmount'])
train_set['TotalIncome']= train_set['ApplicantIncome'] +train_set['CoapplicantIncome']
train_set['TotalIncome_log']=np.log(train_set['TotalIncome'])

train_set.describeribe()

sns.distplot(train_set.ApplicantIncome,kde=False)

from sklearn.preprocessing import LabelEncoder
category= ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
encoder= LabelEncoder()
for i in category:
    train_set[i] = encoder.fit_transform(train_set[i])
    train_set.dtypes

#Modelling

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
def classification_model(model, data, predictors, outcome):
    #Fit the model:
    model.fit(data[predictors],data[outcome])

    #Make predictions on training set:
    predictions = model.predict(data[predictors])

    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

outcome_var= 'Loan_Status'
model_DT = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model_DT, df,predictor_var,outcome_var)

model_RT = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
                 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
                 'LoanAmount_log','TotalIncome_log']
classification_model(model_RT, train,predictor_var,outcome_var)




