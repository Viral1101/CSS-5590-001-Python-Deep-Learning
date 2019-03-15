import pandas
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import LabelEncoder

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("diabetes.csv", header=None).values
bc_dataset = pd.read_csv("Breas Cancer.csv", header=1)
# print(dataset)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(15, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)

# my_second_nn = Sequential() # create model
# my_second_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
# my_second_nn.add(Dense(15, input_dim=8, activation='relu')) # hidden layer
# my_second_nn.add(Dense(1, activation='sigmoid')) # output layer
# my_second_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# my_second_nn_fitted = my_second_nn.fit(X_train, Y_train, epochs=100, verbose=0,
#                                      initial_epoch=0)

input1 = Input(shape=(8,))
hidden1 = Dense(20, activation='relu')(input1)
hidden2 = Dense(15, activation='relu')(hidden1)
output1 = Dense(1, activation='sigmoid')(hidden2)
nn = Model(inputs=input1, outputs=output1)

nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_fitted = nn.fit(X_train, Y_train, epochs=100, verbose=0, initial_epoch=0)

bc_dataset = bc_dataset.apply(LabelEncoder().fit_transform)
bc_dataset = bc_dataset.values
bc_X_train, bc_X_test, bc_Y_train, bc_Y_test = train_test_split(bc_dataset[:,2:], bc_dataset[:,1], test_size=0.25,
                                                                random_state=87)

bc_input = Input(shape=(30,))
bc_hidden1 = Dense(80, activation='relu')(bc_input)
bc_hidden2 = Dense(40, activation='relu')(bc_hidden1)
bc_hidden3 = Dense(20, activation='relu')(bc_hidden2)
bc_output = Dense(1, activation='sigmoid')(bc_hidden3)
bc_model = Model(inputs=bc_input, outputs=bc_output)
bc_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
bc_fitted = bc_model.fit(bc_X_train, bc_Y_train, epochs=100, verbose=0, initial_epoch=0)


print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))

# print(my_second_nn.summary())
# print(my_second_nn.evaluate(X_test, Y_test, verbose=0))

print(nn.summary())
print(nn.evaluate(X_test, Y_test, verbose=0))

print(bc_model.summary())
print(bc_model.evaluate(bc_X_test, bc_Y_test, verbose=0))
