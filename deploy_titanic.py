import numpy as np
import pandas as pd 
import pickle

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("data/train.csv")

train_data['Age'] = train_data['Age'].interpolate()

y = train_data["Survived"]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = pd.get_dummies(train_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

print("Saving the model...")
with open("deploy_titanic_classifier.pkl","wb") as f:
    pickle.dump(model,f)
print("Model saved!")