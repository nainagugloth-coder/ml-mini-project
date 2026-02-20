import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("data.csv")

# Separate features and target
X = data[["hours"]]
y = data["marks"]

# Split into training and testing data (real ML practice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)

# Model evaluation
print("Model Performance")
print("------------------")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# User input prediction
study_hours = float(input("\nEnter number of study hours: "))
predicted_marks = model.predict([[study_hours]])

print(f"Predicted marks for {study_hours} hours study: {predicted_marks[0]:.2f}")
