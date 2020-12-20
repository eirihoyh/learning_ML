#  We will be looking at words used in movie reviews to evaluate if a movie is good or bad
from tensorflow.keras import datasets, preprocessing, Sequential, layers
import numpy as np

data = datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(
    num_words=88_000)

word_index = data.get_word_index()

word_index = {key: (value + 3) for key, value in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([value, key] for (key, value) in word_index.items())

train_data = preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],
                                                  padding="post", maxlen=250)
test_data = preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"],
                                                 padding="post", maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


#  model down here

model = Sequential()
model.add(layers.Embedding(88_000, 16))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10_000]
x_train = train_data[10_000:]

y_val = train_labels[:10_000]
y_train = train_labels[10_000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512,
                     validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)


test_review = test_data[0]
predict = model.predict([test_review])
print(f"Review: \n{decode_review(test_review)}\nPrediction: {str(predict[0])}\nActual: "
      f"{str(test_labels[0])}")
print(results)

model.save("../data/model.h5")
