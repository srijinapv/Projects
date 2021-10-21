import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

data_forest = pd.read_csv("forestfire.csv")

#checking null values
data_forest.isnull().sum()

data_forest.describe()
data_forest.shape

#get the datatypes of features
data_forest.dtypes

imp_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points']

#histogram
feature_set.hist(figsize=(13, 13))
plt.show()

#get the boxplot
box_hill_3pm = data_forest.boxplot('Hillshade_3pm')

#Corralation between parameters
plt.figure(figsize=(12, 8))
corr = feature_set.corr()
sb.heatmap(corr, annot=True)
plt.show()

#separate features and target
final_feature = data_forest.iloc[:, :54]
y = data_forest.iloc[:, 54]

#Split the data into test and train formate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(final_feature, y, test_size=0.25, random_state=10)

#Model building
#Random Forest
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100)

#fit
RFC.fit(X_train, y_train)

#prediction
y_pred = RFC.predict(X_test)

#score
print("Accuracy is ", RFC.score(X_test, y_test)*100)


