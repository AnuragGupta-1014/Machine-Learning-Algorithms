import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('carprices.csv')

print(df.head(4))

# First we will train the upper 8 row and then test last 2 row
plt.xlabel('mileage')
plt.ylabel('Sell Price')
plt.scatter(df['Mileage'],df['Sell Price($)'])
plt.show()

x = df[['Mileage','Age(yrs)']]
y = df['Sell Price($)']

print(x)
  
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2)   
 #we can also add random_state = 10  for not choosing random mileage each time

print('Size of trainig data',len(X_train), ' Size of test data',len(X_test))


from sklearn.linear_model import LinearRegression


reg = LinearRegression()

reg.fit(X_train,Y_train)
print(reg)

print(reg.predict(X_test))
print(reg.score(X_test,Y_test))            # Gives Model accuracy