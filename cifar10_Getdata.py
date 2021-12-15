from tensorflow.keras.datasets import cifar10
from pathlib import Path
from PIL import Image

#データを(train_images,train_labels)と(test_images, test_labels)に格納
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

#保存先場所をoutput_dirに設定
output_dir = Path("cifar10/testt")

for i,(img, label) in enumerate(zip(test_images, test_labels)):
    save_dir = output_dir / str(label[0])
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / f"{i}.jpg"

    img = Image.fromarray(img)
    img.save(save_path)

"""
#データをディレクトリーに保存
for i, (img, label) in enumerate(zip(train_images, train_labels)):
    save_dir = output_dir / str(label[0])
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / f"{i}.jpg"

    img = Image.fromarray(img)
    img.save(save_path)
"""