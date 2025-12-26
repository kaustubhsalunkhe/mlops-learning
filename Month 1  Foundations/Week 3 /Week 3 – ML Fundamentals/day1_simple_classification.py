# Simple rule-based "model"
def predict(score):
    if score >= 50:
        return 1
    else:
        return 0

scores = [30, 55, 80, 45]

predictions = [predict(s) for s in scores]

print("Scores:", scores)
print("Predictions:", predictions)

