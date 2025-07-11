import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.activations import *
import pandas as pds
import xml.etree.ElementTree as ET

from PIL import Image
from IPython.display import display
from pathlib import Path
import json

def process_image(image_path):
  # Function based on Google Gemini output
  # Search terms: 'PIL convert image to RGB' and 'PIL resize image'
  img = Image.open(file_path)

  new_size = (400, 600)
  resized_image = img.resize(new_size)
  img = resized_image.convert("L")
  # No save?
  # resized_image.save(file_path)

  # Convert to numpy array for TensorFlow
  img_array = np.array(img)

  return img_array

def process_image_noresize(image_path):
  # Function based on Google Gemini output
  # Search terms: 'PIL convert image to RGB' and 'PIL resize image'
  img = Image.open(file_path)

  new_size = (400, 600)
  img = img.convert("L")
  ## resized_image.save(file_path)

  # Convert to numpy array for TensorFlow
  img_array = np.array(img)

  return img_array/255

def cnn(input_shape):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Input(shape=input_shape))
  model.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(32, activation='relu'))
  model.add(tf.keras.layers.Dense(16, activation='relu'))
  model.add(tf.keras.layers.Dense(4))
  return model

def mlp(input_shape):
   model = tf.keras.Sequential()
   model.add(tf.keras.layers.Input(shape=input_shape))
   model.add(tf.keras.layers.Flatten())
   model.add(tf.keras.layers.Dense(256, activation='relu'))
   model.add(tf.keras.layers.Dense(128, activation='relu'))
   model.add(tf.keras.layers.Dense(32, activation='relu'))
   model.add(tf.keras.layers.Dense(4))
   return model

def process_JSON():
  with open('python_scripts/boundingbox.json', 'r') as file:
      data = json.load(file)
    
  annot_dict = {}
  for datapoint in data:
     name = datapoint['name']
     annot_dict[name] = (datapoint['x'], datapoint['y'], datapoint['width'], datapoint['height'])
  
  return annot_dict


file_path = "original_scans/Reel169/ViewScan_010000.tif"

input_array = []

def get_train_array(annot_dict):
  with open('python_scripts/boundingbox.json', 'r') as file:
      data = json.load(file)

  names = [key for key in annot_dict]
  train_array = []
  for datapoint in data:
     if datapoint['name'] in names:
        file_path = Path(datapoint["name"])
        img_array = process_image_noresize(file_path)
        train_array.append(img_array)
  
  return train_array

def inference(model):
  directory = Path("original_scans/Reel169")
  predictions = {}
  test_array = []
  for file_path in directory.iterdir():
     if file_path.is_file():
        # print(f"Inference file path: {file_path}")
        img_array = process_image_noresize(file_path)
        # print(f"input shape: {img_array.shape}")
        test_array.append(img_array)
  
  test_array = np.array(test_array)
  # print(f"Test array: {test_array}")
  predictions = model.predict(test_array)
  # print(f"Predictions: {predictions}")
  predictions_list = predictions.tolist()
  #       print(prediction)
  #       predictions[file_path] = prediction
  
  with open('boundingbox_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(predictions_list, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    # Code for loop based on Google Gemini output
    # Search term: 'python loop through files in directory'
    input_shape = None
    directory = Path("original_scans/Reel169")
    for file_path in directory.iterdir():
        if file_path.is_file():
            # print(file_path)
            img_array = process_image_noresize(file_path)
            # print(img_array)
            input_array.append(img_array)
            # print(f"input shape: {img_array.shape}")
            input_shape = img_array.shape
    
    model = cnn((600, 400, 1))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError)

    input_array = np.array(input_array)
    annot_dict = process_JSON()
    train_array = np.array(get_train_array(annot_dict))
    Y = np.zeros((len(annot_dict), 4))
    # print(Y)
    for i, key in enumerate(annot_dict):
      Y[i] = annot_dict[key]
    print(input_array.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_array, Y))
    train_dataset = train_dataset.batch(10)
    history = model.fit(train_dataset, epochs=100)

    inference(model)