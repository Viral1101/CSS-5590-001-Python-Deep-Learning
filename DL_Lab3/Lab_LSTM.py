import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# Initialize settings for TensorBoard
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)

train = pd.read_csv('./train.tsv', delimiter='\t', encoding='utf-8')
test = pd.read_csv('./test.tsv', delimiter='\t', encoding='utf-8')

train = train[['Phrase', 'Sentiment']]
test = test[['Phrase']]

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer2 = Tokenizer(num_words=max_features, split=' ')

tokenizer.fit_on_texts(train['Phrase'].values)
tokenizer2.fit_on_texts(test['Phrase'].values)

X = tokenizer.texts_to_sequences(train['Phrase'].values)
X = pad_sequences(X)

X2 = tokenizer2.texts_to_sequences(test['Phrase'].values)
X2 = pad_sequences(X2, maxlen=len(X[0]))

print(len(X[0]))
print(len(X2[0]))

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train['Sentiment'])
y_train = to_categorical(integer_encoded)

# y_train = np_utils.to_categorical(train['Sentiment'])
num_classes = y_train.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y_train, test_size=0.33, random_state=seed)

epochs = 10
batch = 30
embed_dim = 128
lstm_out = 196


def create_model():
    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model


# model.fit(X_train, Y_train, epochs = epochs, batch_size=batch, verbose = 1)
# score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch)
# print(score)
# print(acc)

# model.save('./model2' + '.h5')
#

model2 = KerasClassifier(build_fn=create_model)
epochs2 = [1, 2, 3]
batch2 = [10, 20, 30]
param_grid = dict(batch_size=batch2, epochs=epochs2)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model2, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
