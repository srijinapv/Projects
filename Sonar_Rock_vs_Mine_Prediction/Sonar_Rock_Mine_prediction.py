# Prediction of Sonar Rock Vs Mine using Logistic regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data_sonar = pd.read_csv("sonar_data.csv")
data_sonar.shape
data_sonar.describe()

data_sonar[60].value_counts()
data_sonar.groupby(60).mean()


X_sonar = data_sonar.drop(columns = 60,axis = 1)
#Label column
Y_sonar = data_sonar[60]

#spliting the data into train and test
X_train, X_test,Y_train, Y_test = train_test_split(X_sonar,Y_sonar ,test_size = 0.2 , stratify = Y_sonar ,random_state = 1)

model_sonar = LogisticRegression()
#training the logistic regression model with train data
model_sonar.fit(X_train,Y_train)

#Accuracy on training data
X_train_prediction = model_sonar.predict(X_train)
train_accuracy = accuracy_score(X_train_prediction, Y_train)
print(train_accuracy)

#Accuracy on test data
X_test_prediction = model_sonar.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction, Y_test)
print(test_accuracy)

# Predictive Model
user_input_for_prediction = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)
#convert the user input to numpy array
user_data_numpy_array = np.asarray(user_input_for_prediction)
#reshape the user data to one instance
user_data_reshaped = user_data_numpy_array.reshape(1,-1)
prediction = model_sonar.predict(user_data_reshaped)
if prediction == 'R':
    print("prediction is Rock")
else:
    print("Prediction is Mine ")