import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Sentiment.csv', encoding="latin1")
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

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
print(X)
X = pad_sequences(X)
print(X)
embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
# print(model.summary())

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# batch_size = 32


# model = createmodel()
model = KerasClassifier(build_fn=createmodel)

batch_size = [30, 40]
epochs = [1, 2]
param_grid= dict(batch_size=batch_size, epochs=epochs)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model2 = createmodel()
model2.fit(X_train, Y_train, epochs = grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'], verbose = 2)
score,acc = model2.evaluate(X_test,Y_test,verbose=2,batch_size=grid_result.best_params_['batch_size'])
print(score)
print(acc)

# model.save('./model' + '.h5')


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

result = model2.predict_classes(X)

# labelencoder = LabelEncoder()
# integer_encoded = labelencoder.fit_transform(data['sentiment'])

print(result)
print(text)
print(labelencoder.inverse_transform(result))
