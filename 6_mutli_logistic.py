import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits    

digits = load_digits()
print(dir(digits))

print(digits.data[0])

plt.gray()
for i in range(3):
    plt.matshow(digits.images[i])                                     #have 1797 imgs
    plt.show()

print(digits.target[0:5])

#                           NOW WE ARE TRAIN THE MODEL ON  digits.target and digits.data
from sklearn.model_selection import train_test_split


X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target,test_size=0.2)

print('train data size: ', len(X_train),'test data size: ', len(X_test),"total is : ",len(X_train)+len(X_test))



from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train,Y_train)
print(model.score(X_test,Y_test))




# Prediction

print(model.predict([digits.data[12]]))
print(model.predict([digits.data[2]]))
print(model.predict([digits.data[32]]))
print(model.predict(digits.data[7:19]))