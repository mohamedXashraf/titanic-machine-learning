import DataPreprocessor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# --------------------------------------- Model --------------------------------------- #
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(12),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics="accuracy"
)


# --------------------------------------- Model Training --------------------------------------- #
train_data = pd.read_csv("Titanic/titanic_train.csv")
train_data = DataPreprocessor.process(train_data)
X, y = train_data.drop("Survived", axis=1), train_data['Survived']
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

# Train
model.fit(X_train, y_train, epochs=200)

# Prediction
predictions = model.predict(X_cv)

for i in range(len(y_cv)):
    prediction = 0
    if predictions[i][0] >= 0.5:
        prediction = 1
    print("prediction: {} ==> actual: {}".format(prediction, y_cv.iloc[i]))

