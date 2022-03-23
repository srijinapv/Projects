#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

#loading dataset
data = pd.read_csv("wc-at.csv")

data.shape # get number of rows and coloumns
data.describe()

plt.bar(height = data.AT, x = np.arange(1, 110, 1))
plt.hist(data.AT) #histogram
plt.boxplot(data.AT) #boxplot
plt.bar(height = data.Waist, x = np.arange(1, 110, 1))
plt.hist(data.Waist) #histogram
plt.boxplot(data.Waist) #boxplot

# Scatter plot
plt.scatter(x = data['Waist'], y = data['AT'], color = 'green')

# correlation
np.corrcoef(data.Waist, data.AT)

#calculating covariance

cov_output = np.cov(data.Waist, data.AT)[0, 1]
cov_output

# Simple Linear Regression
model = smf.ols('AT ~ Waist', data = data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(data['Waist']))

#Regression line
plt.scatter(data.Waist,data.AT)
plt.plot(data.Waist, pred1, "b")

#Error calculation
res1 = data.AT - pred1
res1_sqr = res1 * res1
mse1 = np.mean(res1_sqr)
rmse1 = np.sqrt(mse1)
print(f'RMSE = {rmse1}')

sm.qqplot(data, line ='45')
plt.show()

# Data is not normal so need to transform the data
#Model building on transformed data
#Log transformation
# X = log(waist) Y = AT

plt.scatter(x = np.log(data['Waist']), y = data['AT'], color = 'red')
np.corrcoef(np.log(data.Waist), data.AT) #correlation

model2 = smf.ols('AT ~ np.log(Waist)', data = data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(data['Waist']))

# Regression Line
plt.scatter(np.log(data.Waist), data.AT)
plt.plot(np.log(data.Waist), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = data.AT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = data['Waist'], y = np.log(data['AT']), color = 'green')
np.corrcoef(data.Waist, np.log(data.AT)) #correlation

model3 = smf.ols('np.log(AT) ~ Waist', data = data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(data['Waist']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(data.Waist, np.log(data.AT))
plt.plot(data.Waist, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = data.AT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = data).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(data))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(data.Waist, np.log(data.AT))
plt.plot(X, pred4, color = 'blue')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = data.AT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


#choose the best model using RMSE
data1 = {"MODEL":pd.Series(["SLR","Log Model", "Exp Model", "Poly Model"]), "RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse = pd.DataFrame(data1)
table_rmse

#Best model
from sklearn.model_selection import train_test_split
train ,test = train_test_split(data, test_size = 0.2)
finalmodel = smf.ols('np.log(AT)~ Waist + I( Waist*Waist)', data = train).fit()
finalmodel.summary()

#Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_AT = np.exp(test_pred)
pred_test_AT

#Model Evaluation on Test data
test_res = test.AT - pred_test_AT
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_AT = np.exp(train_pred)
pred_train_AT

# Model Evaluation on train data
train_res = train.AT - pred_train_AT
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

