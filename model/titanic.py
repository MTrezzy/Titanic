import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

train_data['Age'] = train_data['Age'].interpolate()
test_data['Age'] = test_data['Age'].interpolate()
test_data['Fare'] = test_data['Fare'].interpolate()

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")