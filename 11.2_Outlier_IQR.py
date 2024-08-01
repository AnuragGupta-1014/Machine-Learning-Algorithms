import pandas as pd 

df = pd.read_csv('heights.csv')

print(df.shape,df.describe())

# Using IQR 
'''
   It sperate the data in 4 equal part(0.25, 0.50, 0.75, 1.00)
   and outlirer are determined by (0.75-0.25) val
'''
Q1 = df.height.quantile(0.25)
Q3 = df.height.quantile(0.75)
print("min height :",Q1,"Max height :",Q3)             

# Simply Q3-Q1(0.75-0.25)
IQR = Q3-Q1

lower = Q1-1.5*IQR
upper = Q3+1.5*IQR
print(lower,upper)       # outlier


# again simply as 3-std 
new_df = (df[(df.height<upper)&(df.height>lower)])      # Data without outlier
print(new_df)
print(new_df.shape)