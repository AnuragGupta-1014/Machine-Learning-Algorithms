import pandas as pd 

df = pd.read_csv('titanic.csv')
print(df.head())
print(df.columns)
df.drop(['PassengerId','Name','SibSp','Parch','Ticket', 'Cabin', 'Embarked'],axis='columns',inplace=True)
print(df.head())


target = df.Survived
inputs = df.drop(['Survived'],axis='columns')

dummies =pd.get_dummies(inputs.Sex)         # Asign the int 0 and 1 
print(dummies.head(2))


inputs = pd.concat([inputs,dummies],axis='columns')
print(inputs.head(2))

inputs.drop('Sex',axis='columns',inplace=True)
print("==----------------------------------------------===")
print(inputs.columns[inputs.isna().any()])              #age column hav null value

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.head(2))


# Now train 

from sklearn.model_selection import train_test_split


X_train,X_test,Y_train,Y_test = train_test_split(inputs,target,test_size=0.2)

print(len(X_train),'                 ',len(X_test))



from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))

print(model.predict(X_test[:10]))
print(model.predict_proba(X_test[:10]))
print(model.predict(X_test[10:20]))