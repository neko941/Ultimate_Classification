from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

def mtVGG16(input_shape=(224, 224, 3), output_units=5):
    # Generate the model
    model = Sequential(layers=None, name='mtVGG16')

    # Block 1
    # Layer 1: Convolutional
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1'))
    # Layer 2: Convolutional
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='block1_conv2'))
    # Layer 3: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    # Layer 4: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_conv1'))
    # Layer 5: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='block2_conv2'))
    # Layer 6: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    # Layer 7: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv1'))
    # Layer 8: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv2'))
    # Layer 9: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu', name='block3_conv3'))
    # Layer 10: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    # Layer 11: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv1'))
    # Layer 12: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv2'))
    # Layer 13: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block4_conv3'))
    # Layer 14: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    # Layer 15: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv1'))
    # Layer 16: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv2'))
    # Layer 17: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu', name='block5_conv3'))
    # Layer 18: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool'))

    # Layer 19: Flatten
    model.add(Flatten())
    # Layer 20: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 21: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 22: Softmax Layer
    model.add(Dense(units=output_units, activation='softmax'))

    return model