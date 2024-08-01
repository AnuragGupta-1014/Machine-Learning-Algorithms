import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model


import pickle

df = pd.read_csv("price.csv")
print(df)

plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.Area,df.Price,marker='+')
plt.show()

           # For Pridecting the value of some area ,we will train our model by linear Reg

reg = linear_model.LinearRegression()
reg.fit(df[['Area']],df.Price)               # Area is X axis and Y is Price X is always in[["area"]]
print(reg.predict([[4000]]),reg.predict([[3300]]))

#OR
area = np.array([[4000]])  # Input should be a 2D array
# print(reg.predict(area))

print('------------------------------------')
print("m is :",reg.coef_,'x is :',area,'b is :',reg.intercept_)          #Y = m*x+b 

p = reg.predict(df[['Area']])
df['onemore'] = p                                       # Add new column in org

df.to_csv('new.csv')
print(df.Area.median())

# with open('model_pickle','wb') as f:             # it will save file for further use
#     pickle.dump(reg,f)


    #or

# import joblib
# joblib.dump(reg,'model_Joblib')

# # using it directly for predictions

# load = joblib.load('model_Joblib')
# print("By using the Joblib :",load.predict([[1200]]))