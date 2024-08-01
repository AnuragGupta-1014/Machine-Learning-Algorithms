import pandas as pd 

df = pd.read_csv('spam.csv')
print(df.head(3))

a = df.groupby('Category').describe()             # Category is column name
print(a)

#        Firstly we have to convert the data on 0 an 1 for model building

df['spam'] = df['Category'].apply(lambda x: 1 if x =='spam' else 0)     # if the data is spam then 1 otherwise 0
print(df.head())



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(df.Message,df.spam,test_size=0.25)


# now for msg we have to identify repeated keywords and count them, for this--

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_Count = v.fit_transform(X_train.values)
print(X_train_Count.toarray()[:4])


# using naive bayes 

from sklearn.naive_bayes import MultinomialNB

model =MultinomialNB()

model.fit(X_train_Count,Y_train)         # Model Always take int[0,1] for fitting process

#                 TAKING SOME EXAMPLE

emails =[
    '1: Hey mohan, can we get togther to watch the football',
    '2: upto 20% discount on parking,exclusive offer just now for you. Dont miss the reward',
    'want to join the club now,your friend ani have it'
]

emails_count = v.transform(emails)
print(model.predict(emails_count))

X_test_Count =v.transform(X_test)
print("Through Transform",model.score(X_test_Count,Y_test))


#             For remove the transfomr step we use pipeline

from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

clf.fit(X_train,Y_train)              #now we don't need to transfrom
print("Through Pipeline :",clf.score(X_test,Y_test)) 

print(clf.predict(['hello Gautam, i want you as my head coach']))
print(clf.predict(['hello Gautam, want to grab the offer of 20% discount on shoes']))