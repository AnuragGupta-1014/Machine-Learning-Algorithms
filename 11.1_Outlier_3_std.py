import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('heightsss.csv')
print(df.shape)
print(df.describe())

# Creating the histogram
plt.hist(df.height,bins=20,rwidth=0.6)            # bins are no. of bar
plt.xlabel('Height(inches)')
plt.xlabel('Numbers of objects')
# plt.show()

print("Minimun height :",df.height.min())
print("Maximum height :",df.height.max())


#using scipy            for bell curve

from scipy.stats import norm 
import numpy as np 

plt.hist(df.height, bins=20, rwidth=0.6, density=True, alpha=0.6, color='g')  # Normalized histogram
plt.xlabel('Height (inches)')
plt.ylabel('Density')
plt.title('Height Distribution with Bell Curve')

# Define the range for the bell curve
rng = np.linspace(df.height.min(), df.height.max(), 100)
plt.plot(rng, norm.pdf(rng, df.height.mean(), df.height.std()), color='r')
# plt.show()


# finding the outliers                 Using 3 std deviation

upper = df.height.mean() + 3*df.height.std()
lower = df.height.mean() - 3*df.height.std()
print(upper,lower)

print(df[(df.height>upper)|(df.height<lower)])        # printing the upper and lower data than limit(outlier)
print(df[(df.height>upper)|(df.height<lower)].shape)    

new_df = (df[(df.height<upper)&(df.height>lower)])      # Data without outlier
print(new_df)