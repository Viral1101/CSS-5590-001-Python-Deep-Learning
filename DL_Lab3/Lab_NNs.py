import numpy
import keras
import pandas as pd
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier

# Initialize settings for TensorBoard
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Set random seed for reproducibility
seed = 42
numpy.random.seed(seed)

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

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(train['Sentiment'])
y_train = to_categorical(integer_encoded)

# y_train = np_utils.to_categorical(train['Sentiment'])
num_classes = y_train.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y_train, test_size=0.33, random_state=seed)

embed_dim = 128

print(X_train.shape[1])

epochs = 10
batch = 100
lrate = 0.01
decay = lrate/epochs

def create_model():
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
    model.add(Conv1D(128, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


model = create_model()

#
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch, callbacks=[tbCallBack])
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
# model.save('./model' + '.h5')

# model2 = KerasClassifier(build_fn=create_model)
#
# epochs2 = [10, 15]
# batch2 = [30, 50, 100]
# param_grid = dict(batch_size=batch2, epochs=epochs2)
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(estimator=model2, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X_train, Y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
