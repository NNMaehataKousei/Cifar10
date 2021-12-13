import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

def main():
    #データのダウンロード
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    #データの準備
    train_image_generator = ImageDataGenerator(rescale=1./255) # 学習データのジェネレータ
    validation_image_generator = ImageDataGenerator(rescale=1./255) # 検証データのジェネレータ

    model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(32, 32 ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.summary()

    hist = model.fit(x_train, y_train,
    batch_size=32, epochs=50,
    verbose=1,
    validation_data=(x_test, y_test))


    
                                        
if __name__ == "__main__":
    main()