import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data.csv")

X = data[["hours"]]
y = data["marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
hours = [[5]]
prediction = model.predict(hours)

print("Predicted marks for 5 hours study:", prediction[0])
