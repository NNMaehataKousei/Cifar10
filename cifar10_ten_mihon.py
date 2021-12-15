import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import dlt
import os

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
from keras.callbacks import EarlyStopping

# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
data = dlt.cifar.load_cifar10()

# preprocess the data in a suitable way
# reshape the image matrices to vectors
#RGB 255 = white, 0 = black
X_train = data.train_images.reshape([-1, 32, 32, 3])
X_test = data.test_images.reshape([-1, 32, 32, 3])
print('%i training samples' % X_train.shape[0])
print('%i test samples' % X_test.shape[0])
print(X_train.shape)

# convert integer RGB values (0-255) to float values (0-1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# convert class labels to one-hot encodings
Y_train = to_categorical(data.train_labels, 10)
Y_test = to_categorical(data.test_labels, 10)

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

data_augmentation = tensorflow.keras.Sequential([
  tensorflow.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tensorflow.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
#分類ヘッドを作る

inputs = tensorflow.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(10, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
loss='categorical_crossentropy',
optimizer=Adam(lr=0.001),
metrics=['accuracy'])

log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard"
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

fit = model.fit(X_train, Y_train,
              batch_size=128,
              epochs=20, #shouldn't be raised to 100, because the overfitting occurs.
              verbose=2,
              validation_split=0.1,
              callbacks=tensorboard_callback   )

score = model.evaluate(X_test, Y_test,
                    verbose=0
                    )
print('Test score:', score[0])
print('Test accuracy:', score[1])

# ----------------------------------------------
# Some plots
# ----------------------------------------------

# make output directory
folder = 'results'
if not os.path.exists(folder):
    os.makedirs(folder)
    
model.save(os.path.join(folder, 'my_model_tl.h5'))

