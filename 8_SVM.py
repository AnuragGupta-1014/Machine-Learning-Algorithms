import pandas as pd 
from sklearn.datasets import load_iris

iris = load_iris()
print(dir(iris))
# print(iris.feature_names,)
for i in iris.feature_names:
    print(i)

# Converting into Dataframe
    
df = pd.DataFrame(iris.data,columns=iris.feature_names)

print(df.shape)
print(df.head())
print(iris.target_names)     # three type 

# Append / add new column target

df['target'] =iris.target
print(df.head())
a = df[df.target==1].head()
print(a)

# creating new column

df['Flower name'] = df.target.apply(lambda x : iris.target_names[x])    #assign the iris flower to Flower name column  
a['Flower name'] = a.target.apply(lambda x : iris.target_names[x])    #assign the iris flower to Flower name column  
print(df.head(2))
print(a.head(2))


# visualization

import matplotlib.pyplot as plt

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color= 'red',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color= 'blue')
# plt.show()


# training the model

from sklearn.model_selection import train_test_split

X = df.drop(['target','Flower name'],axis='columns')
print(X.head())

Y = df.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
print(len(X_train),len(X_test))


#                 NOW APPLYING THE SVC(support vector machine)
from sklearn.svm import SVC

model =SVC()
model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))
