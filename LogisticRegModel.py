import pandas as pd
import DataPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

model = LogisticRegression()

# --------------------------------------- Model Training --------------------------------------- #
train_data = pd.read_csv("Titanic/titanic_train.csv")
train_data = DataPreprocessor.process(train_data)
X, y = train_data.drop("Survived", axis=1), train_data['Survived']
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

# Train
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_cv)

# Evaluation
print(classification_report(y_cv, predictions))


# --------------------------------------- Model Testing --------------------------------------- #
test_data = pd.read_csv("Titanic/titanic_test.csv")
test_data.dropna(subset="Fare", inplace=True)
test_data = DataPreprocessor.process(test_data)

predictions = model.predict(test_data)
print(predictions)
