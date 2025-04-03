import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset (Features: Age, Income, Loan Amount | Target: Credit Score)
data = {
    "Age": [25, 45, 35, 50, 23, 40, 60, 30, 48, 55],
    "Income": [30000, 60000, 50000, 80000, 25000, 70000, 90000, 40000, 75000, 85000],
    "LoanAmount": [5000, 20000, 15000, 25000, 4000, 18000, 30000, 10000, 22000, 27000],
    "CreditScore": [1, 1, 1, 0, 1, 1, 0, 1, 0, 0]  # 1 = Good, 0 = Bad
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into features (X) and target (y)
X = df.iloc[:, :-1]  # Features
y = df["CreditScore"]  # Target labels

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling (Normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Compute confusion matrix and accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)

# Classify a new sample
new_sample = np.array([[32, 45000, 12000]])  # Example input
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
print("\nNew Sample Prediction (Credit Score):", "Good" if prediction[0] == 1 else "Bad")
