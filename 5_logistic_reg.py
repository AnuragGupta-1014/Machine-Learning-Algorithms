import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('insurance_data.csv')
print(df.head(4))
print(df.shape)

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
# plt.show()




# for train and test set 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df[['age']],df.bought_insurance,test_size= 0.1)
print(X_test)

    
# Preformed the logistic reg

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,Y_train)

print(model.predict(X_test))
print(model.score(X_test,Y_test))    # if 1 then model is perfect ,if 0 model is not perfect
print(model.predict_proba(X_test))      # predict the % of buying the insurance

print(model.predict([[24]]))           # passing the random age
print(model.predict([[94]]))           # passing the random age

# n =int(input("Enter the age :"))
# print(model.predict([[n]]))


# Assuming model is already defined and trained
while True:
    user_input = input("Enter the age (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the loop.")
        break
    try:
        n = int(user_input)
        prediction = model.predict([[n]])
        print(f"Prediction: {prediction}")
    except ValueError:
        print("Invalid input. Please enter a valid integer or 'exit' to quit.")
