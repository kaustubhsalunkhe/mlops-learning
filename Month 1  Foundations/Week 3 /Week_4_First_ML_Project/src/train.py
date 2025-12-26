import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


# -------------------
# Paths
# -------------------
BASE_DIR = "/home/mladmin/mlops-learning/Month 1  Foundations/Week 3 /Week_4_First_ML_Project/"

DATA_PATH = os.path.join(BASE_DIR, "Data", "dataset.csv")   # <-- CSV FILE
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -------------------
# Ensure model directory exists
# -------------------
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------
# Load data
# -------------------
data = pd.read_csv(DATA_PATH)

X = data[['age', 'salary']]
y = data['buy']

# -------------------
# Train-test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------
# Scaling
# -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# Train model
# -------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# -------------------
# Save artifacts
# -------------------
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("✅ Training completed. Model and scaler saved.")


# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

