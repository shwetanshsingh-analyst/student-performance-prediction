# Student Performance Prediction System

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Simple dataset create kar rahe hain
data = {
    "study_hours": [2, 3, 4, 5, 6, 7, 8, 1, 2, 9],
    "attendance": [50, 60, 65, 70, 75, 80, 90, 40, 45, 95],
    "previous_score": [35, 40, 50, 55, 60, 70, 85, 30, 32, 90],
    "result": [0, 0, 1, 1, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# 2. Input aur output select karna
X = df[["study_hours", "attendance", "previous_score"]]
y = df["result"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model train karna
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Accuracy check
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 7. New student prediction
new_student = pd.DataFrame({
    "study_hours": [5],
    "attendance": [75],
    "previous_score": [60]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")