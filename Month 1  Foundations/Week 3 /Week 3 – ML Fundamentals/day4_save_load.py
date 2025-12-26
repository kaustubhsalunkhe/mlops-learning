from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

#DataSet
X = [[30],[45],[60],[75],[90]]
y = [0,0,1,1,1]

#Split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)

#Train
model = LogisticRegression()
model.fit(X_train, y_train)

#Save Model
joblib.dump(model,'model.pkl')
print('Model Saved as Model.pkl')

#Load Model 
loaded_model = joblib.load('model.pkl')

#Predict
predictions = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print('Predictions:',predictions)
print('Accuracy:',accuracy)
