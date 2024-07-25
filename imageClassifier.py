import keras
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.src.metrics import Precision, Recall, BinaryAccuracy
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, RandomFlip, RandomContrast, RandomBrightness
import cv2
import os

"""gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.list_physical_devices('GPU')"""
dataset = tf.keras.utils.image_dataset_from_directory("data", labels="inferred")
data_iterator = dataset.as_numpy_iterator()
batch = data_iterator.next()
"""fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])"""
dataset = dataset.map(lambda x, y: (x / 255, y))
dataset.as_numpy_iterator().next()
data_size = len(np.concatenate([i for x, i in dataset], axis=0))
train_size = int(data_size*1)
val_size = int(data_size*0)
test_size = 0
print(data_size, train_size, val_size, test_size)
train = dataset.take(train_size)
val = dataset.skip(train_size).take(val_size)
test = dataset.skip(train_size+val_size)
model = Sequential([
    # Preprocessing
    # RandomFlip('horizontal'),  # Flip left-to-right
    # RandomContrast(0.2),  # Contrast change by up to 20%
    # RandomBrightness(0.25)  # Change brightness up to +-25%
    Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(keras.optimizers.Adam(learning_rate=5e-5), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(dataset, epochs=200, callbacks=[tensorboard_callback])
model.save("chord_classifier_model.keras")
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
"""pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)"""
"""img = cv2.imread('chord_one.jfif')
plt.imshow(img)
plt.show()
resize = tf.image.resize(img, (256, 256))
yhat = model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5:
    print(f'Am')
else:
    print(f'C')"""

