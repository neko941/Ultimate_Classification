from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pathlib import Path
import os

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def callback(name, patience=100, save_dir='result'):
    checkpoint_best = ModelCheckpoint(filepath=Path(save_dir) / "weights" / f"{name}_best.h5",
                                      save_best_only=True,
                                      save_weights_only=False,
                                      verbose=0)

    checkpoint_last = ModelCheckpoint(filepath=Path(save_dir) / "weights" / f"{name}_last.h5",
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