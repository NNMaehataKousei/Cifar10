#必要なパッケージのインポート
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

#データダウンロード
data = dlt.cifar.load_cifar10()

#データの前処理：画像行列をベクトルに再形成
X_train = data.train_images.reshape([-1, 32, 32, 3])
X_test = data.test_images.reshape([-1, 32, 32, 3])
print('%i training samples' % X_train.shape[0])
print('%i test samples' % X_test.shape[0])
print(X_train.shape)

#データの値は0~255となっている。計算効率のため0~1にスケーリングする。
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

#クラスラベル
Y_train = to_categorical(data.train_labels, 10)
Y_test = to_categorical(data.test_labels, 10)

#次に層を重ねていくが、Xceptionの構造を使うので、すべての層を通過した後のモデルのインスタンスをbase_modelとして取り出します。
base_model = VGG16(weights='imagenet', include_top=False)

#ネットワーク構造を固定する
for layer in base_model.layers:
    layer.trainable = False

#データ増強
data_augmentation = tensorflow.keras.Sequential([
  tensorflow.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tensorflow.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

#分類ヘッドを作り新たな重みを決める
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

#モデルを作成
model = Model(inputs=base_model.input, outputs=predictions)

#モデルのコンパイル
model.compile(
loss='categorical_crossentropy',
optimizer=Adam(lr=0.0005),
metrics=['accuracy'])

#学習経過を可視化するための設定
log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/cifar10"
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#モデルの学習
fit = model.fit(X_train, Y_train,
              batch_size=128,
              epochs=50, #shouldn't be raised to 100, because the overfitting occurs.
              verbose=2,
              validation_split=0.1,
              callbacks=tensorboard_callback   )

#評価関数
score = model.evaluate(X_test, Y_test,
                    verbose=0
                    )
print('Test score:', score[0])
print('Test accuracy:', score[1])

# make output directory
folder = 'results'
if not os.path.exists(folder):
    os.makedirs(folder)
    
model.save(os.path.join(folder, 'my_model_tl.h5'))


preds = model.predict(X_test)

# plot some test images along with the prediction
for i in range(100):
    dlt.utils.plot_prediction(
        preds[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname=os.path.join(folder, 'test-%i.png' % i))

