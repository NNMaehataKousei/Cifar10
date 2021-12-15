

from matplotlib.colors import from_levels_and_colors
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import keras

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.ops.control_flow_ops import from_control_flow_context_def
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add_eager_fallback, tensor_strided_slice_update
from tensorflow.python.ops.gen_math_ops import less_eager_fallback
from tensorflow.python.training.tracking import base

from tensorflow.keras.datasets import cifar10

#データの前処理
#データのダウンロード
#(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()

#前処理
BATCH_SIZE = 128
IMG_SIZE = (32, 32)

PATH = "C:/Users/nf_maehata/Desktop/転移学習/cifar10/cifar10"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "test")
train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

"""
元のデータセットにはテストセットが含まれていないので、テストセットを作成します。
作成には、tf.data.experimental.cardinality を使用して検証セットで利用可能なデータのバッチ数を調べ、
そのうちの 20％ をテストセットに移動します。
"""
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print("Number of Validation batches: %d" % tf.data.experimental.cardinality(validation_dataset))
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

#パフォーマンスのためにデータセットを構成する

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#データ増強を使用する
#ただし、model.fitを呼び出したときのみアクティブになる。モデルがmodel.evaulateやmodel.fitなどの推論モードで使用する場合はアクティブしない
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

def tenni_gakushu(base):
        
    #別の方法でRescalingレイヤーを使用して、ピクセル値を[0,255]から[-1,1]にリスケールすることも可能である
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

    #事前トレーニング済み畳み込みニューラルネットワークから基本モデルを作成する。
    #MoblieNet V2モデルから基本モデルを生成する。

    #ImageNetでトレーニングした重みで事前に読み込んだMobileV2モデルをインスタンス化する。引数はinclude_top=False
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model_1 = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_3 = tf.keras.applications.MobileNetV3Small(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    ##tensorboardで記録する
    if base == 1:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v1"
        base_model = base_model_1
        #ピクセル値をリスケールする
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    
    elif base == 2:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v2"
        base_model = base_model_2
        #ピクセル値をリスケールする
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    elif base == 3:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v3"
        base_model = base_model_3
        #ピクセル値をリスケールする
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    #この特徴抽出器は32x32x3の画像を5x5x1280の特徴ブロックに変換する。これで画像のバッチ例がどうなるかを見てみましょう。
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    #特徴を抽出する。
    #畳み込みベースを凍結させる。
    base_model.trainable = False
    base_model.summary()

    #分類ヘッドを追加する
    #特徴ブロッカら予測値を生成する。そのためにレイヤーを使って5x5空間の空間位置を平均化し、特徴画像ごとの1280要素ベクトルに変換させる。
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    #画像ごとに単一の予測値に変換する
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    #Keras Functional APIを使用して、データ増強、リスケール、base_model、特徴抽出レイヤーを凍結してモデルを構築させる。
    inputs = tf.keras.Input(shape=(32,32,3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs,outputs)

    #モデルをコンパイルする
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=["accuracy"])

    #モデル概要
    model.summary()

    len(model.trainable_variables)

    #モデルをトレーニングする
    initial_epochs = 10
    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}" .format(loss0))
    print("initial accuracy: {:.2f}" .format(accuracy0))

    history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset, callbacks=tensorboard_callback)

    #学習結果を曲線で可視化
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    #plt.ylim([min(plt.ylim()),1])
    plt.ylim(0.8,1.0)
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


    #######################################################################
    #ファインチューニング

    #パフォーマンスをさらに向上させるために、追加した分類器のトレーニングと平行して、事前トレーニング済みモデルの最上位レイヤーの重みをトレーニング
    #するというもの。ただし事前トレーニング済みモデルをトレーニング不可に設定し、最上位の分類気をトレーニングした後に行うようにする。

    #モデルの最上位レイヤーを解凍する
    base_model.trainable = True

    print("Number of layers in the bass model: ", len(base_model.layers))

    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    #モデルをコンパイルする
    """
    かなり大規模なモデルをトレーニングしているため、事前トレーニング済みの重みを再適用する場合は、
    この段階では低い学習率を使用することが重要です。そうしなければ、モデルがすぐに過適合を起こす可能性があります。
    """
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10), metrics=["accuracy"])
    model.summary()

    len(model.trainable_variables)


    #モデルのトレーニングを続ける
    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs
    history_fine = model.fit(
        train_dataset,epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
        callbacks=tensorboard_callback
        )

    acc += history_fine.history["accuracy"]
    val_acc += history_fine.history["val_accuracy"]

    loss += history_fine.history["loss"]
    val_loss += history_fine.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    #評価と予測をする
    loss, accuracy = model.evaluate(test_dataset)
    print("Test accuracy :", accuracy)

    ###############################################################################
    #Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.axis("off")

tenni_gakushu(1)