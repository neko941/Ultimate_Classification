import os
import sys
import argparse
import numpy as np
from pathlib import Path

from keras.metrics import AUC
from keras.metrics import Recall
from keras.optimizers import Adam
from keras.metrics import Precision
from keras.metrics import TrueNegatives
from keras.metrics import TruePositives
from keras.metrics import FalsePositives
from keras.metrics import FalseNegatives
from keras.metrics import BinaryAccuracy

from keras.preprocessing.image import ImageDataGenerator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import display
from utils.general import callback

from utils.metrics import F1_score

from models.VGG16 import mtVGG16
from keras.applications import VGG16 as tVGG16 

def train(model, save_name, train_set, val_set, epochs=50, learning_rate=0.0001):
    model.compile(loss = 'categorical_crossentropy',
                  optimizer=Adam(learning_rate),
                  metrics=[
                      BinaryAccuracy(name='accuracy'),
                      TruePositives(name='tp'),
                      FalsePositives(name='fp'),
                      TrueNegatives(name='tn'),
                      FalseNegatives(name='fn'), 
                      Precision(name='precision'),
                      Recall(name='recall'),
                      AUC(name='auc'),
                      AUC(name='prc', curve='PR'),
                      F1_score
                      ])
    
    model._name = save_name
    model.summary()      
    
    history = model.fit(train_set,
                        validation_data=val_set,
                        epochs=epochs,
                        callbacks=callback(save_name),
                        verbose=1)
    # Plotting Accuracy, val_accuracy, loss, val_loss
    display(history=history,
            save_name=f'{save_name}_result.png')
    
    # Predict Data Test
    pred = model.predict(val_set)
    pred = np.argmax(pred,axis=1)
    labels = (train_set.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    
    print('\033[01m              Classification_report \033[0m')
    
    print('\033[01m              Results \033[0m')
    # Results
    results = model.evaluate(val_set, verbose=0)
    print("    Test Loss:\033[31m \033[01m {:.5f} \033[30m \033[0m".format(results[0]))
    print("Test Accuracy:\033[32m \033[01m {:.2f} \033[30m \033[0m".format(results[1] * 100))
    
    return results

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--overwrite', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')

    parser.add_argument('--mtVGG16', action='store_true', help='Using my tensorflow VGG16 model')
    parser.add_argument('--tVGG16', action='store_true', help='Using tensorflow VGG16 model')

    return parser.parse_known_args()[0] if known else parser.parse_args()



def main(opt):
    DIRECTORY = "00.Animals"
    CLASSES = [dir for root, directories, files in os.walk(DIRECTORY) for dir in directories]
    BATCH = opt.batch_size # 128
    IMGSZ = (256, 256)

    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1.0/255.0,
        preprocessing_function=None,
        data_format=None,
        dtype=None,
        validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(directory=DIRECTORY,
                                                        target_size=IMGSZ,
                                                        batch_size=BATCH,
                                                        class_mode='categorical',
                                                        interpolation="nearest",
                                                        subset="training",
                                                        classes=CLASSES)
    test_generator = train_datagen.flow_from_directory(directory=DIRECTORY,
                                                    target_size=IMGSZ,
                                                    batch_size=BATCH,
                                                    class_mode='categorical',
                                                    interpolation="nearest",
                                                    subset="validation",
                                                    classes=CLASSES)
    if opt.mtVGG16:
        mtVGG16_model = mtVGG16(input_shape=IMGSZ+(3,), output_units=len(CLASSES))

        result_mtVGG16 = train(model=mtVGG16_model,
                            train_set=train_generator,
                            val_set=test_generator,
                            epochs=opt.epochs,
                            save_name='mtVGG16', 
                            learning_rate=0.0001)
    
    if opt.tVGG16:
        tVGG16_model = tVGG16(input_shape=IMGSZ+(3,), output_units=len(CLASSES))

        result_tVGG16 = train(model=tVGG16_model,
                            train_set=train_generator,
                            val_set=test_generator,
                            epochs=opt.epochs,
                            save_name='tVGG16', 
                            learning_rate=0.0001)

def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)