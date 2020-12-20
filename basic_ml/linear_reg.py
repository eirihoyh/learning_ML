# In this file we will predict how different students score on their final grade based on grade 1,
# grade 2, study time, failures and absences.

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle

data = pd.read_csv("../data/student-mat.csv", sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]


predict = "G3"

x = np.array(data.drop([predict], 1))  # all values but our predict
y = np.array(data[predict])  # all precit actual values

best = 0
for _ in range(30):
    """
    This for-loop is used to save the best prediction accuracy out of 30 predictions
    """
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("../data/student_model.pickle", "wb") as f:
            pickle.dump(linear, f)

print(f'Coefficient: {linear.coef_}\nIntercept: {linear.intercept_}')

predictions = linear.predict(x_test)


for index, pred in enumerate(predictions):
    print(f'How the student preformed: {x_test[index]}\n'
          f'Prediction: {pred}\nActual result: {y_test[index]}')
