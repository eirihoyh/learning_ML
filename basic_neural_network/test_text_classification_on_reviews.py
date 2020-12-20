from tensorflow.keras import datasets, models, preprocessing

data = datasets.imdb

word_index = data.get_word_index()

word_index = {key: (value + 3) for key, value in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

model = models.load_model("../data/model.h5")


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


with open("../data/lion_king_review.txt") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"],
                                             padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])