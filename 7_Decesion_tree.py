import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('salaries.csv')

print(df.head(4))
print("----------------------------------------")
input = df.drop('salary_more_then_100k',axis='columns')           # X train
target = df['salary_more_then_100k']                         # Y train

print(input.head(4),'\n',target.head(4))

from sklearn.preprocessing import LabelEncoder

company = LabelEncoder()
job = LabelEncoder()
degree = LabelEncoder()

input['com_'] = company.fit_transform(input['company'])
input['job_'] = company.fit_transform(input['job'])
input['degree_'] = company.fit_transform(input['degree'])

print(input.head(4))

input_n = input.drop(['company','job','degree'],axis='columns')       # Bcoz we need only no. for prediction
print(input_n.head(5))






# Now training 

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(input_n,target)

print(model.score(input_n,target))
print(model.predict([[2,2,1]]))      # 2,2,1 is form the google, job ,degree in no.