# Simple dataset
data = [
    (30, 0),
    (45, 0),
    (60, 1),
    (75, 1),
    (90, 1)
]

# Split manually
train_data = data[:3]
test_data = data[3:]

def predict(score):
    return 1 if score >= 50 else 0

# Test model
correct = 0
for score, label in test_data:
    pred = predict(score)
    if pred == label:
        correct += 1

accuracy = correct / len(test_data)
print("Accuracy:", accuracy)
