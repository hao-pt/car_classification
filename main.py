import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
import time
import os
import pydot
from IPython.display import SVG
# from tensorflow.keras.utils.vis_utils import model_to_dot
import argparse
import cv2
from matplotlib import pyplot as plt
import json_tricks
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['Chevrolet Orlando', 'Ford Ranger', 'Honda Civic', 'Honda CR-V', 'Hyundai Accent',\
     'Hyundai Grand i10', 'Kia Cerato', 'Kia Morning', 'Mazda 3', 'Mercedes-Benz C', 'Mitsubishi Xpander',\
     'Nissan Terra', 'Toyota Fortuner', 'Toyota Innova', 'Toyota Vios']

def get_modelid_and_logid_path(model_dir, log_dir):
    # Model name: datetime object containing current date and time
    # dd/mm/YY H:M:S
    timeid = time.strftime("%Y_%m_%d-%H_%M_%S")
    modelid = "model_" + timeid + ".h5"
    logid = "run_" + timeid
    return os.path.join(model_dir, modelid), os.path.join(log_dir, logid)

def classification_layers(base_model, keep_prob, no_categories):
    model = Sequential()
    model.add(base_model)
    # The pros of GlobalAveragePooling2D are more efficient to reduce the number of parameters than FC layer and smaller number of features.
    # GlobalAveragePooling2D convert from (batch_size, height, width, channels) -> (batch_size, channels)
    model.add(GlobalAveragePooling2D(name='gap2d'))
    model.add(Dropout(keep_prob, name="dr")) # Prevent overfitting
    model.add(Dense(no_categories, activation='softmax', name="softmax"))

    return model

def make_dir(path):
    try:
        # if the path is already existed, so this function will surpass it
        os.makedirs(path, exist_ok=True)
        print("Directory '%s' was created successfully" %path)
    except OSError as error:
        print("Directory '%s' can not be created" %path) 

def load_data(data_dir, anot_file, img_dimension, test_size=0.25, seed=174):
    """
    Load dataset from data_dir and anot_file

    Args:
        data_dir: Directory of dataset
        anot_file: Anotation file which is a dictionary contains "boxes", "scores" and "scaled_area" for each image
        img_dimension: dimension of image that is a tuple of (height, width)
    Returns:
        (X_train, Y_train): train dataset
        (X_test, Y_test): test dataset        
        
        Where
            X: image data with shape (#images, height, width, 3)
            Y: one-hot vector with shape (#images, C)
    """
    height, width = img_dimension
    # Load anotation file
    with open(anot_file, "r") as json_file:
        anot_data = json_tricks.load(json_file)

    X = []    
    Y = []
    for parent in os.listdir(data_dir):
        if "." in parent:
            continue
        parent_dir = os.path.join(data_dir, parent)
        for img_file in os.listdir(parent_dir):
            if img_file.startswith(".") or img_file.startswith("Icon"):
                continue
            img = cv2.imread(os.path.join(parent_dir, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get bounding boxes
            boxes = anot_data[os.path.join(parent, img_file)]['boxes']
            
            for box in boxes:
                # lower left and higher right points
                y1, x1, y2, x2 = box
                # Get the region of interest (car object)
                roi = img[y1:y2, x1:x2]

                # # Resize image
                # roi = center_crop_and_resize(roi, image_size=height)
                # print(roi.shape)
                roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_AREA) #cv2.INTER_NEAREST)
                
                # Append img and label
                X.append(roi)
                Y.append(CLASS_NAMES.index(parent))

                # plt.figure()
                # plt.imshow(roi)
    
    # plt.show()

    X = np.array(X)
    Y = np.array(Y).reshape((-1, 1))
    # Convert to one-hot
    Y = to_categorical(Y)
    
    # Split dataset with stratified fashion
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    return (X_train, Y_train), (X_test, Y_test)

def data_augmentation(X_train, Y_train, X_test, Y_test, batch_size):
    """
    Data augmentation

    Args:
        X_train: training data with shape (#images, height, width, 3)
        Y_train: trainging labels (one-hot form) with shape (#images, 15)
        X_test: test data with shape (#images, height, width, 3)
        Y_test: test labels (one-hot form) with shape (#images, 15)
        batch_size: Batch size
    
    Returns:
        train_generator: training data with augmentation
        test_generator: test data with scale by 1./255
    """
    # Get image dimension
    height, width = X_train.shape[1:-1]
    
    # Init ImageDataGenerator object
    train_datagen = ImageDataGenerator(rescale=1./255, \
                                        shear_range=0.15, \
                                        zoom_range=0.15, \
                                        horizontal_flip=True, \
                                        rotation_range=30, \
                                        width_shift_range=0.2, \
                                        height_shift_range=0.2, \
                                        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    # fit parameters from data
    train_datagen.fit(X_train)
    test_datagen.fit(X_test)

    # Generate data into batch
    train_generator = train_datagen.flow(X_train, Y_train, \
                                    batch_size=batch_size, \
                                    shuffle=False)

    test_generator = test_datagen.flow(X_test, Y_test, \
                                    batch_size=batch_size, shuffle=False)

    return train_generator, test_generator


def main(args):

    # Define hyper-params
    height, width = args.dimension
    no_epochs = args.no_epochs
    lr = args.learning_rate
    keep_prob = args.keep_prob
    batch_size = args.batch
    log_dir = args.log_dir # log_dir = os.path.join(os.curdir, "logs")
    model_dir = args.model_dir
    data_dir = args.data_dir
    anot_file = args.anot_file

    # Make dir if there are not existed
    make_dir(log_dir)
    make_dir(model_dir)
    
    modelid_path, logid_path = get_modelid_and_logid_path(model_dir, log_dir) # Generate modelid and logid
    
    # Load data

    input_shape = (height, width, 3)
    no_train_examples = 2000
    no_test_examples = 1000
    
    no_categories = len(CLASS_NAMES)
    step_per_epoch = no_train_examples//batch_size
    val_steps = no_test_examples//batch_size

    # Load data
    (X_train, Y_train), (X_val, Y_val) = load_data(data_dir, anot_file, (height, width))


    print("Number of examples: %d" %(Y_train.shape[0]+Y_val.shape[0]))
    print("X_train shape: {}".format(X_train.shape))
    print("Y_train shape: {}".format(Y_train.shape))
    print("X_val shape: {}".format(X_val.shape))
    print("Y_val shape: {}".format(Y_val.shape))

    # Data augmentation
    train_data, val_data = data_augmentation(X_train, Y_train, X_val, Y_val, batch_size)

    # Step 1: Freeze and Pretrain

    # Load pretrained model as base model
    # include_top=False mean that we dont take the final FC of original dataset
    base_model = EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=False) 

    # Freeze the base_model to avoid destroying the pretrained weights
    base_model.trainable = False

    # Add our top classification layers onto base_model to classify our own dataset
    model = classification_layers(base_model, keep_prob, no_categories)
    print(model.summary())

    sgd = optimizers.SGD(learning_rate=lr, momentum=0.9, decay=0.01)
    model.compile(loss="categorical_crossentropy",
                optimizer=sgd,
                metrics=["accuracy"])
    

    # Using callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(modelid_path, save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    tensorboard_cb = keras.callbacks.TensorBoard(logid_path)

    # Fit model
    history = model.fit_generator(train_data, \
                            steps_per_epoch=step_per_epoch, \
                            epochs=no_epochs, \
                            validation_data=val_data, \
                            validation_steps= val_steps, \
                            verbose=1, \
                            use_multiprocessing=True, \
                            workers=4, \
                            callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

    # Save model
    model.save(modelid_path)

    if args.plot_learning_curve:
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    # Step 2: Fine-tuning the pretrained weights
    # base_model.trainable = True
    # plot_model(base_model, to_file='base_model.png', show_shapes=True)
    # SVG(model_to_dot(base_model).create(prog='dot', format='svg'))
    # for layer in base_model.layers:
    #     print(layer.name)

def ParseArgs():
    parser = argparse.ArgumentParser(description="Car classification")
    parser.add_argument("-ddir", "--data_dir", type=str, required=True,\
         help="Data directory")
    parser.add_argument("-wdir", "--weights_dir", type=str, default="./car_model.h5",\
         help="Path to pretrained weights of model")
    parser.add_argument("-mdir", "--model_dir", type=str, default="./logs",\
         help="Directory to save trained model (*.h5 file)")
    parser.add_argument("-ldir", "--log_dir", type=str, default="./models",\
         help="Directory to write logs and save files")
    parser.add_argument("-af", "--anot_file", type=str, default="",\
         help="Anotation file of car dataset (a dictionary includes 'boxes', 'scores' and 'scaled_area')")
    parser.add_argument("-dim", "--dimension", type=int, nargs=2, default=[150, 150],\
         help="Dimension of input images are fed into model")
    parser.add_argument("-e", "--no_epochs", type=int, default=20,\
         help="Number of epochs to run")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,\
         help="Learning rate")
    parser.add_argument("-kp", "--keep_prob", type=float, default=0.2,\
         help="Keep probability was added on the top layer")
    parser.add_argument("-b", "--batch", type=int, default=32,\
         help="Keep probability was added on the top layer")
    parser.add_argument("-pl", "--plot_learning_curve", action='store_true',\
         help="Plot learning curve")


    return parser.parse_args()

if __name__ == "__main__":
    main(ParseArgs())

