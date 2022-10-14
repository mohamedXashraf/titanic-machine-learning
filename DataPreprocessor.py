import pandas as pd

def process(data):
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    sex = pd.get_dummies(data["Sex"])
    embark = pd.get_dummies(data["Embarked"])
    pclass = pd.get_dummies(data["Pclass"])
    data.drop(["Sex", "Embarked", "Pclass"], axis=1, inplace=True)
    data = pd.concat([data, sex, embark, pclass], axis=1)
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    return data
