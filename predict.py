
import pandas as pd
import joblib

model = joblib.load("model.joblib")

# Sample input
sample = pd.DataFrame([{
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 192776,
    "education": "Bachelors",
    "educational-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "gender": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States"
}])

prediction = model.predict(sample)
print("Predicted Income:", prediction[0])
