# KNN stands for K-Nearest Neighbors
# KNN is a machine learning algorithm used for classifying data. Rather than coming up with a
# numerical prediction such as a students grade or stock price it attempts to classify data into
# certain categories. In the next few tutorials we will be using this algorithm to classify cars
# in 4 categories based upon certain features.

import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("../data/car.data", sep=",")
print(data.head())

# since all coloumns contains words and not integers, we use preprocessing.LabelEncoder to make
# sklearn choose numbers for us. Working with integer will make it a lot easier
# further in the process

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
print(buying)

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(f"Accuracy: {acc}")

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]  # names of the different classes

for index, pred in enumerate(predicted):
    print(f"Predicted: {names[pred]}, Data: {x_test[index]}, Actual: {names[y_test[index]]}")
    distance, neighbor = model.kneighbors([x_test[index]], 9)
    print(f"Neighbors: {neighbor}\nDistance: {distance}")
