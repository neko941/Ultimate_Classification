from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

def stVGG16(input_shape=(224, 224, 3), output_units=5):
    # Generate the model
    model = Sequential()
    # Layer 1: Convolutional
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    # Layer 2: Convolutional
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 3: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 4: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 5: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 6: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 7: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 8: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 9: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 10: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 11: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 12: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 13: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 14: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 15: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 16: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 17: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 18: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 19: Flatten
    model.add(Flatten())
    # Layer 20: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 21: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 22: Softmax Layer
    model.add(Dense(units=output_units, activation='softmax'))

    return model