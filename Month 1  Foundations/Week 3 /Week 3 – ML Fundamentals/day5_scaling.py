from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

#Dataset With Different Scales
X = [
    [25, 20000],
    [30, 30000],
    [35, 40000],
    [40, 60000],
    [45, 80000]
]
y = [0, 0, 0, 1, 1]

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


#Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#Predict
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print('Accuracy:',accuracy)

#Save both Model and Scaler
joblib.dump(model,'model.pkl')
joblib.dump(scaler,'Scaler.pkl')

