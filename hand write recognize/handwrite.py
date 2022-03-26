# 2022/03/24 20:36
# have a good day!
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.layers import AveragePooling2D
from keras.layers import Flatten
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    file = np.load('mnist.npz')
    x_train = file.f.x_train
    y_train = file.f.y_train
    x_test = file.f.x_test
    y_test = file.f.y_test
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    x_train = (x_train.reshape(60000, 28, 28, 1) - 255.0) / -255.0
    x_test = (x_test.reshape(10000, 28, 28, 1) - 255.0) / -255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=2000, batch_size=4096)

    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss:', loss)
    print('accuracy', accuracy)
    model.save('手写体训练模型/model_2000.h5')



