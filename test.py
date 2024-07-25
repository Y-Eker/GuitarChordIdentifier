import cv2
import numpy as np
import keras
import tensorflow as tf


model = keras.models.load_model("chord_classifier_model.keras")

img = cv2.imread(f"chord_two.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resize = tf.image.resize(img, (256, 256))
input_img = np.expand_dims(resize / 255, 0)
yhat = model.predict(input_img)
print(yhat)
print(input_img.shape)
if yhat > 0.5:
    print(f'C')
else:
    print(f'Am')
