import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

housing=pd.read_csv("/Housing.csv")
housing.head()
#Check the datatypes
housing.dtypes
housing.describe()
#get the size
housing.shape
#checking for null values
housing.isnull().sum()
#histogram of price
plt.hist(x = housing['price'])

#count plot of furnsing status
count_furnished = sns.countplot(x = 'furnishingstatus', data = housing)
count_road = sns.countplot(x = 'mainroad', data = housing)
count_guest = sns.countplot(x = 'guestroom', data = housing)
count_basement = sns.countplot(x = 'basement', data = housing)

#boxplot to check presence of outliers
sns.boxplot(x = housing['price'])
sns.boxplot(x = housing['area'])
sns.boxplot(x = housing['bedrooms'])
sns.boxplot(x = housing['parking'])

#Removal of outliers
IQR = housing['price'].quantile(0.75)-housing['price'].quantile(0.25)
upper = housing['price'].quantile(0.75) + (IQR*1.5)
print(upper)
lower = housing['price'].quantile(0.25) - (IQR*1.5)
print(lower)
new_df = np.where(housing['price']>upper, True,np.where(housing['price']<lower , True, False))
removed_df = housing.loc[~(new_df),]

IQR = housing['area'].quantile(0.75)-housing['area'].quantile(0.25)
upper = housing['area'].quantile(0.75) + (IQR*1.5)
print(upper)
lower = housing['area'].quantile(0.25) - (IQR*1.5)
print(lower)
new_df = np.where(housing['area']>upper, True,np.where(housing['area']<lower , True, False))
removed_df = housing.loc[~(new_df),]

IQR = housing['bedrooms'].quantile(0.75)-housing['bedrooms'].quantile(0.25)
upper = housing['bedrooms'].quantile(0.75) + (IQR*1.5)
print(upper)
lower = housing['bedrooms'].quantile(0.25) - (IQR*1.5)
print(lower)
new_df = np.where(housing['bedrooms']>upper, True,np.where(housing['bedrooms']<lower , True, False))
removed_df = housing.loc[~(new_df),]

IQR = housing['stories'].quantile(0.75)-housing['stories'].quantile(0.25)
upper = housing['stories'].quantile(0.75) + (IQR*1.5)
print(upper)
lower = housing['stories'].quantile(0.25) - (IQR*1.5)
print(lower)
new_df = np.where(housing['stories']>upper, True,np.where(housing['stories']<lower , True, False))
removed_df = housing.loc[~(new_df),]

IQR = housing['parking'].quantile(0.75)-housing['parking'].quantile(0.25)
upper = housing['parking'].quantile(0.75) + (IQR*1.5)
print(upper)
lower = housing['parking'].quantile(0.25) - (IQR*1.5)
print(lower)
new_df = np.where(housing['parking']>upper, True,np.where(housing['parking']<lower , True, False))
removed_df = housing.loc[~(new_df),]

IQR = housing['bathrooms'].quantile(0.75)-housing['bathrooms'].quantile(0.25)
upper = housing['bathrooms'].quantile(0.75) + (IQR*1.5)
print(upper)
lower = housing['bathrooms'].quantile(0.25) - (IQR*1.5)
print(lower)
new_df = np.where(housing['bathrooms']>upper, True,np.where(housing['bathrooms']<lower , True, False))
removed_df = housing.loc[~(new_df),]


#converting catogorical column to numerical
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
removed_df['mainroad'] = labelencoder.fit_transform(removed_df['mainroad'])
removed_df['guestroom'] = labelencoder.fit_transform(removed_df['guestroom'])
removed_df['basement'] = labelencoder.fit_transform(removed_df['basement'])
removed_df['hotwaterheating'] = labelencoder.fit_transform(removed_df['hotwaterheating'])
removed_df['airconditioning'] = labelencoder.fit_transform(removed_df['airconditioning'])
removed_df['prefarea'] = labelencoder.fit_transform(removed_df['prefarea'])
removed_df['furnishingstatus'] = labelencoder.fit_transform(removed_df['furnishingstatus'])

#assigning dependent variable and independent variable
X = removed_df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
                'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                'parking', 'prefarea', 'furnishingstatus']]

y = removed_df['price']

#Model Building, chosen linear regression
from sklearn.model_selection import train_test_split

#spliting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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
print('R_squared:', metrics.r2_score(y_test, predictions))