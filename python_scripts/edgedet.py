import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.activations import *
import pandas as pds

from PIL import Image
from IPython.display import display
from pathlib import Path
import json
import re

def process_image_noresize(image_path):
  # Function based on Google Gemini output
  # Search terms: 'PIL convert image to RGB' and 'PIL resize image'
  img = Image.open(image_path)

  new_size = (1944, 2592)
  img = img.convert("L")
  ## resized_image.save(file_path)

  # Convert to numpy array for TensorFlow
  img_array = np.array(img)

  return img_array/255

def get_images(path, exclusions, limit=500):
  directory = Path(path)
  input_array = []
  names = []
  i = 0
  for file_path in directory.iterdir():
    if (i == limit):
      break

    if file_path.is_file():
      img_array = process_image_noresize(file_path)
      input_array.append(img_array)
      i += 1
  
  return input_array

def cnn(input_shape):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Input(shape=input_shape))
  model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(4))
  return model

def process_JSON():
  with open('python_scripts/boundingbox.json', 'r') as file:
      data = json.load(file)
    
  annot_dict = {}
  for datapoint in data:
     name = datapoint['name']
     name_path = Path(name)
     annot_dict[str(name_path)] = (datapoint['x'], datapoint['y'], datapoint['width'], datapoint['height'])
  
  # print(annot_dict)
  return annot_dict

def get_train_data(annot_dict, path):
  img_array = []
  directory = Path(path)
  names = [key for key in annot_dict]
  # print(names)
  for file_path in directory.iterdir():
    # print(str(file_path))
    if file_path.is_file() and str(file_path) in names:
      print(file_path)
      img = process_image_noresize(file_path)
      img_array.append(img)
  
  return img_array

def inference(model, input):
  input = np.array(input)
  predictions = model.predict(input)
  prediction_list = predictions.tolist()

  with open('boundingbox_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(prediction_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
  path = "original_scans/Reel169"
  input_array = get_images(path, [], limit = 30)

  model = cnn((1944, 2592, 1))
  model.summary()
  model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError)

  annot_dict = process_JSON()
  train_data = np.array(get_train_data(annot_dict, path))
  Y = np.zeros((len(annot_dict), 4))
  for i, key in enumerate(annot_dict):
      Y[i] = annot_dict[key]
  
  # print(train_data)
  train_dataset = tf.data.Dataset.from_tensor_slices((train_data, Y))
  train_dataset = train_dataset.batch(3)
  history = model.fit(train_dataset, epochs=20)

  inference(model, input_array)