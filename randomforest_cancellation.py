import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load the Iris dataset from scikit-learn
df = pd.read_csv('customer_data.csv')
inputForRun = pd.read_csv('customer_data_input.csv')

count_vectorizer1 = CountVectorizer(decode_error='ignore')
X_gender = count_vectorizer1.fit_transform(df['gender'])

count_vectorizer2 = CountVectorizer(decode_error='ignore')
X_company = count_vectorizer2.fit_transform(df['company'])

count_vectorizer3 = CountVectorizer(decode_error='ignore')
X_productCode = count_vectorizer3.fit_transform(df['productCode'])

count_vectorizer4 = CountVectorizer(decode_error='ignore')
X_occupation = count_vectorizer4.fit_transform(df['occupation'])

count_vectorizer5 = CountVectorizer(decode_error='ignore')
X_smoker = count_vectorizer5.fit_transform(df['smoker'])

count_vectorizer6 = CountVectorizer(decode_error='ignore')
X_paymentType = count_vectorizer6.fit_transform(df['paymentType'])

X_other = df[['salary', 'age', 'term', 'premium', 'sumassured']]
X = pd.concat([pd.DataFrame(X_gender.toarray()), 
               pd.DataFrame(X_company.toarray()), pd.DataFrame(X_productCode.toarray()), 
               pd.DataFrame(X_occupation.toarray()), pd.DataFrame(X_smoker.toarray()), 
               pd.DataFrame(X_paymentType.toarray()), X_other], axis=1)
X.columns = X.columns.astype(str)
y = df['policyStatus']
print(X)
inputToPredict_gender = count_vectorizer1.transform(["male"])
inputToPredict_company = count_vectorizer2.transform(["ENDICIL"])
inputToPredict_productCode = count_vectorizer3.transform(["BBB"])
inputToPredict_occupation = count_vectorizer4.transform(["AA"])
inputToPredict_smoker = count_vectorizer5.transform(["No"])
inputToPredict_paymentType = count_vectorizer6.transform(["DD"])

inputToPredict_other = [[3819,42,17,92,55845]]
inputToPredict = pd.concat([pd.DataFrame(inputToPredict_gender.toarray()), 
               pd.DataFrame(inputToPredict_company.toarray()), pd.DataFrame(inputToPredict_productCode.toarray()), 
               pd.DataFrame(inputToPredict_occupation.toarray()), pd.DataFrame(inputToPredict_smoker.toarray()), 
               pd.DataFrame(inputToPredict_paymentType.toarray()), pd.DataFrame(inputToPredict_other)], axis=1)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fitting the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate accuracy and print the confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

with open('customer_cancellation_prediction.pkl', 'wb') as f:
    pickle.dump((rf_model,
                       X_other.columns), f)
    
with open('customer_cancellation_prediction_1.pkl', 'wb') as f:
    pickle.dump(count_vectorizer1, f)

with open('customer_cancellation_prediction_2.pkl', 'wb') as f:
    pickle.dump(count_vectorizer2, f)

with open('customer_cancellation_prediction_3.pkl', 'wb') as f:
    pickle.dump(count_vectorizer3, f)

with open('customer_cancellation_prediction_4.pkl', 'wb') as f:
    pickle.dump(count_vectorizer4, f)

with open('customer_cancellation_prediction_5.pkl', 'wb') as f:
    pickle.dump(count_vectorizer5, f)

with open('customer_cancellation_prediction_6.pkl', 'wb') as f:
    pickle.dump(count_vectorizer6, f)
    
print(inputToPredict.shape[1])
print(inputToPredict)
print (rf_model.predict(inputToPredict))
