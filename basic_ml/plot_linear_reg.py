import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import pandas as pd

data = pd.read_csv("../data/student-mat.csv", sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

pickle_in = open("../data/student_model.pickle", "rb")
linear = pickle.load(pickle_in)

p = 'failures'
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel('Finale grade')
plt.show()