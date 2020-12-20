#  In this document we look at different images of cloths and makes a guess of what clothing
#  item that is displayed
from tensorflow.keras import datasets, Sequential, layers
import numpy as np
import matplotlib.pyplot as plt

data = datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0  # transform all values to be between 0 and 1
test_images = test_images/255.0

#  Sequential makes a sequence of layers
model = Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

#  Now we train our model
model.fit(train_images, train_labels, epochs=5)  # Epochs: how many times the same image will be shown

prediction = model.predict(test_images)

for i in range(8):
    plt.grid(False)
    plt.imshow(test_images[i], cmap="binary")
    plt.xlabel(f"Actual: {class_names[test_labels[i]]}")
    plt.title(f"Prediction {class_names[np.argmax(prediction[i])]}")
    plt.show()

