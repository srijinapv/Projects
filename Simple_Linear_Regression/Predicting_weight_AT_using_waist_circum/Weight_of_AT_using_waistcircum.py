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
