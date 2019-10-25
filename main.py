import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPooling2D, Dropout
from tensorflow.keras import Model, Sequential
from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras.preprocessing import image
import time
import os

# Define hyper-params
width = 150
height = 150
no_epoches = 20
no_train_examples = 2000
no_test_examples = 1000
lr = 1e-4
keep_prob = 0.2
batch_size = 64
input_shape = (height, width, 3)
no_categories = 2
step_per_epoch = no_train_examples//batch_size
val_steps = no_test_examples//batch_size


log_dir = os.path.join(os.curdir, "logs")

def get_logid_path(log_dir):
    # Model name: datetime object containing current date and time
    # dd/mm/YY H:M:S
    logid = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(log_dir, logid)

logid_path = get_logid_path(log_dir)


# Data augmentation
train_datagen = image.ImageDataGenerator(rescale=1./255,
                                         shear_range=)

# Step 1: Freeze and Pretrain

# Load pretrained model as base model
base_model = EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=False) # include_top=False mean that we dont take the final FC of original dataset

# Freeze the base_model to avoid destroying the pretrained weights
base_model.trainable = False

# Add our top classification layers onto base_model to classify our own dataset 
model = Sequential()
model.add(base_model)
# The pros of GlobalMaxPooling2D are more efficient to reduce the number of parameters than FC layer and smaller number of features.
model.add(GlobalMaxPooling2D(name='gap2d')) # GlobalMaxPooling2D convert from (batch_size, height, width, channels) -> (batch_size, channels)
model.add(Dropout(keep_prob, name="dr")) # Prevent overfitting
model.add(Dense(no_categories, activation='softmax', name="softmax"))

adam = optimizers.Adam(learning_rate=lr)
model.compile(loss="categorical_crossentropy",
              optimizer=adam,
              metrics=["accuracy"])

print(model.summary())



# Using callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(dtime, save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
tensorboard_cb = keras.callbacks.TensorBoard(logid_path)

# Fit model
history = model.fit_generator(callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

# Save model
model.save("my_model.h5")
# Step 2: Fine-tuning the pretrained weights



