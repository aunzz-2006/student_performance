import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data.csv")

# 🔥 Feature Engineering
df["study_efficiency"] = df["previous"] / (df["hours"] + 1)

# Split
X = df.drop("performance", axis=1)
y = df["performance"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔥 Try Multiple Models
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
lr = LogisticRegression(max_iter=1000)

rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
lr.fit(X_train, y_train)

print("RF Accuracy:", rf.score(X_test, y_test))
print("DT Accuracy:", dt.score(X_test, y_test))
print("LR Accuracy:", lr.score(X_test, y_test))

# 🔥 Hyperparameter Tuning
params = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
grid.fit(X_train, y_train)

model = grid.best_estimator_

# 🔥 Cross Validation
cv_score = cross_val_score(model, X, y, cv=3).mean()
print("Cross Validation Score:", cv_score)

# 🔥 Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model + scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))