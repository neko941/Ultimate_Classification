from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def callback(name, patience=100):
    checkpoint_best = ModelCheckpoint(filepath=f"{name}_best.h5",
                                      save_best_only=True,
                                      save_weights_only=False,
                                      verbose=0)

    checkpoint_last = ModelCheckpoint(filepath=f"{name}_last.h5",
                                      save_best_only=False,
                                      save_weights_only=False,
                                      verbose=0)

    earlystop = EarlyStopping(monitor='val_loss',
                              patience=patience,
                              mode='min',
                              min_delta=0.0001)

    return [checkpoint_best, checkpoint_last, earlystop]

def display(history, save_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, metric in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[metric])
        ax[i].plot(history.history['val_' + metric])
        ax[i].set_title(f'Model {metric}')
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(metric)
        ax[i].legend(['Train', 'Validation'])
    
    plt.show()
    plt.savefig(save_name)