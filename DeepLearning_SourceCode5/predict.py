import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

data = pd.read_csv('Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]
data = data[data.sentiment != 'Neutral']

data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

print(data[data['sentiment'] == 'Positive'].size)
print(data[data['sentiment'] == 'Negative'].size)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])

# text = "Deer in the headlights RT @lizzwinstead: Ben Carson, may be the only brain surgeon who has performed a lobotomy on himself. #GOPDebate"
text = "A lot of good things are happening. We are respected again throughout the world, and that's a great thing."
# text = "@ScottWalker: Didn't catch the full #GOPdebate last night. Here are some of Scott's best lines in 90 seconds. #Walker16 http://t.co/ZSfFâ€¦"
text.lower()
re.sub('[^a-zA-z0-9\s]', '', text)

text = [text]

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(text)
X = tokenizer.texts_to_sequences(text)
print(X)
X = pad_sequences(X, maxlen=28)
print(X)

model = load_model('./model.h5')

result = model.predict_classes(X)

# print(result)
print(text[0])
print(labelencoder.inverse_transform(result)[0])

