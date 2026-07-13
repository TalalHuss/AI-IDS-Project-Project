from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocessing import X_train, X_test, y_train, y_test

# create a model

model = RandomForestClassifier()

# train model

model.fit(X_train, y_train)

# Test model

prediction = model.predict(X_test)

# Print accuracy

print("Accuracy:", accuracy_score(y_test, prediction))
