import pandas as pd
import numpy as np
import os
import pathlib
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

data_dir = pathlib.Path('food/train')
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

train_ds = train_ds.shuffle(100)

data_augmentation = keras.Sequential(
  [
    RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    RandomRotation(0.1),
  ]
)

# Build model
from tensorflow.keras.applications import xception

base_model = tf.keras.applications.Xception(input_shape=[img_height, img_width, 3], include_top=False)
base_model.trainable = False

inputs = keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = xception.preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.summary()

# Train the top layer
model.compile(optimizer = Adam(learning_rate=1e-4),
              loss = SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

checkpoint_filepath = 'model.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[model_checkpoint_callback])

# Do a round of fine-tuning of the entire model
checkpoint_filepath = 'model_finetune.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

base_model.trainable = True
model.summary()

model.compile(optimizer = Adam(learning_rate=1e-5),
              loss = SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[model_checkpoint_callback])

test_dir = pathlib.Path('food/test')

output = pd.DataFrame(columns = ['file', 'prediction'])
model.load_weights('model_finetune.hdf5')

for img_name in os.listdir(test_dir):
  img = tf.keras.utils.load_img(os.path.join(test_dir,img_name), target_size=(img_height, img_width))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  output = output.append({'file': img_name, 'prediction': class_names[np.argmax(predictions)]}, ignore_index=True)

output.to_csv('output.csv', index = False)
