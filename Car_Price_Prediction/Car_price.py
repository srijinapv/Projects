import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("car_price.csv")
df.head()
#Check the datatypes
df.dtypes
df.describe()
#get the size
df.shape
#checking for null values
df.isnull().sum()