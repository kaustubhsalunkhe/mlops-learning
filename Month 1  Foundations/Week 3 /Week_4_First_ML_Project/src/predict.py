import joblib 
import numpy as np

#Load Artifacts
model = joblib.load('/home/mladmin/mlops-learning/Month 1  Foundations/Week 3 /Week_4_First_ML_Project/models/model.pkl')
scaler = joblib.load('/home/mladmin/mlops-learning/Month 1  Foundations/Week 3 /Week_4_First_ML_Project/models/scaler.pkl')

def predict(age, salary):
    features = np.array([[age, salary]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)

    return prediction[0], probability[0][1]

if __name__ == "__main__":
    age = 35
    salary = 50000

    pred, prob = predict(age, salary)
    print("Prediction (0=No, 1=Yes):", pred)
    print("Probability of buying:", round(prob, 2))
