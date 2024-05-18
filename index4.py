import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("insurance.csv")

label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
df['smoker'] = label_encoder.fit_transform(df['smoker'])
df['region'] = label_encoder.fit_transform(df['region'])

train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=42)

train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

model = RandomForestRegressor(random_state=42)

model.fit(train_dataset, train_labels)

mae = mean_absolute_error(test_labels, model.predict(test_dataset))
print("Mean Absolute Error:", mae)

predictions = model.predict(test_dataset)

plt.figure(figsize=(10, 6))
plt.scatter(test_labels, predictions, alpha=0.5)
plt.xlabel('True Expenses')
plt.ylabel('Predicted Expenses')
plt.title('True vs Predicted Expenses')
plt.show()
