import pandas as pd 

df = pd.read_csv('homeprices.csv')
print(df)



#Dumming the coloumn 
Dummies = pd.get_dummies(df.town)
print(Dummies)

merge = pd.concat([df,Dummies],axis = 'columns')
print(merge.head())
