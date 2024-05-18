import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('sms_train.csv')
df_test = pd.read_csv('sms_test.csv')

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df_train['message'], df_train['label'])

predictions = model.predict(df_test['message'])
accuracy = accuracy_score(df_test['label'], predictions)
print("Model Accuracy:", accuracy)

def predict_message(message):
    prediction = model.predict_proba([message])[0]
    likelihood = prediction.max()
    label = 'spam' if prediction.argmax() == 1 else 'ham'
    return [likelihood, label]
