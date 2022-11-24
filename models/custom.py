from keras.layers import Dense

from keras.models import Model

def customize_model(model, input_shape=(128,128, 3), output_units=5):
    model = model(
            input_shape=input_shape,
                    include_top=False,
                    weights='imagenet',
                    pooling='avg')
    model.trainable = False
    inputs = model.input
    x = Dense(64, activation='relu')(model.output)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_units, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model