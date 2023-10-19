import pandas as pd
import pickle

# Load the trained model and feature names from the pickle file
with open('customer_cancellation_prediction.pkl', 'rb') as f:
    random_forest, x_columns = pickle.load(f)

# Load the CountVectorizer object from the pickle file
with open('customer_cancellation_prediction_1.pkl', 'rb') as f:
    count_vectorizer1 = pickle.load(f)

with open('customer_cancellation_prediction_2.pkl', 'rb') as f:
    count_vectorizer2 = pickle.load(f)

with open('customer_cancellation_prediction_3.pkl', 'rb') as f:
    count_vectorizer3 = pickle.load(f)

with open('customer_cancellation_prediction_4.pkl', 'rb') as f:
    count_vectorizer4 = pickle.load(f)

with open('customer_cancellation_prediction_5.pkl', 'rb') as f:
    count_vectorizer5 = pickle.load(f)

with open('customer_cancellation_prediction_6.pkl', 'rb') as f:
    count_vectorizer6 = pickle.load(f)


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

print (random_forest.predict(inputToPredict))
