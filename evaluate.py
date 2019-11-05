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
from efficientnet.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB4
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
import time
import os
import pydot
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
import argparse
import cv2
from matplotlib import pyplot as plt
import json_tricks
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['Chevrolet Orlando', 'Ford Ranger', 'Honda Civic', 'Honda CR-V', 'Hyundai Accent',\
     'Hyundai Grand i10', 'Kia Cerato', 'Kia Morning', 'Mazda 3', 'Mercedes-Benz C', 'Mitsubishi Xpander',\
     'Nissan Terra', 'Toyota Fortuner', 'Toyota Innova', 'Toyota Vios']

def get_logid_path(log_dir):
    # Model name: datetime object containing current date and time
    # dd/mm/YY H:M:S
    timeid = time.strftime("%Y_%m_%d-%H_%M_%S")
    logid = "run_" + timeid
    return os.path.join(log_dir, logid)

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

def load_data(data_dir, anot_file, img_dimension):
    """
    Load dataset from data_dir and anot_file

    Args:
        data_dir: Directory of dataset
        anot_file: Anotation file which is a dictionary contains "boxes", "scores" and "scaled_area" for each image
        img_dimension: dimension of image that is a tuple of (height, width)

    Returns:
        X, Y: dataset       
        
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
    
    return X, Y

def main(args):
    # Define hyper-params
    height, width = args.dimension
    keep_prob = args.keep_prob
    batch_size = args.batch
    log_dir = args.log_dir # log_dir = os.path.join(os.curdir, "logs")
    data_dir = args.data_dir
    anot_file = args.anot_file

    # Make dir if there are not existed
    make_dir(log_dir)

    logid_path = get_logid_path(log_dir) # Generate logid

    print(logid_path)
    make_dir(logid_path)

    # Load data
    X_test, Y_test = load_data(data_dir, anot_file, (height, width))

    no_test_examples = Y_test.shape[0]
    print("Number of examples: %d" %(no_test_examples))
    print("X_test shape: {}".format(X_test.shape))
    print("Y_test shape: {}".format(Y_test.shape))

    input_shape = (height, width, 3)


    no_categories = len(CLASS_NAMES)

    # rescale images
    X_test = X_test / 255.0

    # Load pretrained model as base model
    # include_top=False mean that we dont take the final FC of original dataset
    if args.bn == "B0":
        base_model = EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=False) 
    elif args.bn == "B1":
        base_model = EfficientNetB1(input_shape=input_shape, weights='imagenet', include_top=False) 
    elif args.bn == "B4":
        base_model = EfficientNetB4(input_shape=input_shape, weights='imagenet', include_top=False) 

    base_model.trainable = True
    print(base_model.summary())
    
    # Enable start_layer and its successive layers to be trainable 
    start_layer_to_train = args.start_layer_to_train #"block5a_expand_conv (Conv2D)" #"block4a_expand_conv (Conv2D)" #"block6a_expand_conv (Conv2D)"  
    # Check if using fine-tune
    if start_layer_to_train != "0":
        # If start_layer_to_train == "", we will train the entire network
        is_trainable = False if start_layer_to_train != "" else True
        for layer in base_model.layers:
            if layer.name == start_layer_to_train:
                is_trainable = True
            
            if is_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        base_model.trainable = False

    # Add our top classification layers onto base_model to classify our own dataset
    model = classification_layers(base_model, keep_prob, no_categories)
    print(model.summary())

    # Load pretrained model
    model.load_weights(args.weights_dir)

    # Compile the model
    model.compile(loss="categorical_crossentropy",
                    optimizer="sgd",
                    metrics=["accuracy"])

    # Evaluate 
    # Scalars contain [loss, acc]
    scalars = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
    print(scalars)


    # Save arguments
    with open(os.path.join(logid_path, "revision_infor.txt"), "wt") as f:
        f.write(args.__str__())
        f.write("\n")
        f.write(logid_path)
        f.write("\n")
        f.write(str(scalars))      

def ParseArgs():
    parser = argparse.ArgumentParser(description="Car classification")
    parser.add_argument("-ddir", "--data_dir", type=str, required=True,\
        help="Data directory")
    parser.add_argument("-wdir", "--weights_dir", type=str, default="",\
        help="Path to pretrained weights of model")
    parser.add_argument("-ldir", "--log_dir", type=str, default="./logs",\
        help="Directory to write logs and save files")
    parser.add_argument("-af", "--anot_file", type=str, default="",\
        help="Anotation file of car dataset (a dictionary includes 'boxes', 'scores' and 'scaled_area')")
    parser.add_argument("-dim", "--dimension", type=int, nargs=2, default=[150, 150],\
        help="Dimension of input images are fed into model")
    parser.add_argument("-kp", "--keep_prob", type=float, default=0.2,\
        help="Keep probability was added on the top layer")
    parser.add_argument("-b", "--batch", type=int, default=32,\
        help="Keep probability was added on the top layer")
    parser.add_argument("-pl", "--plot_learning_curve", action='store_true',\
        help="Plot learning curve")
    parser.add_argument("-bn", "--bn", type=str, default= 'B0', choices=["B0", "B1", "B2", "B3", "B4"],\
        help="Model type such as B0, B4.")
    parser.add_argument("-st", "--start_layer_to_train", type=str, default="block5a_expand_conv (Conv2D)",\
        choices=["block5a_expand_conv (Conv2D)", "block4a_expand_conv (Conv2D)", "block6a_expand_conv (Conv2D)", \
            "block7a_se_excite (Multiply)", "0", "", "block7a_expand_conv (Conv2D)"],\
        help="Start layer to fine-tune from EfficientNet model. Notice: for empty string '' means that fine-tune the entire network and '0' means that we dont use fine-tuning at all")

    return parser.parse_args()

# Parse Args
args = ParseArgs()

if __name__ == "__main__":
    main(args)

