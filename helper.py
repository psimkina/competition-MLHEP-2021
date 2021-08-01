import numpy as np 
from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
import tensorflow as tf

def load_data(path): 
    """Upload images from path and returns numpy array of images and corresponding energies."""

    files = [f for f in listdir(path) if isfile(join(path, f))]
    images = []
    energies = []
    
    for k in tqdm(files):
        img = tf.keras.preprocessing.image.load_img(
            path+k, color_mode='grayscale', target_size=(576,576))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])
        images.append(input_arr)
        energies.append(float(re.findall('[0-9]+', k)[4]))
    return np.asarray(images), np.asarray(energies)

def crop_images(images): 
    images = images.reshape(-1,576,576)
    images = images[:, 225:-225, 225:-225]
    return images

def load_test(path): 
    """Upload test images from path and returns numpy array of images and corresponding ids."""
    
    files = [f for f in listdir(path) if isfile(join(path, f))]
    images =[]
    n_id = []
    for k in tqdm(files):
        img = tf.keras.preprocessing.image.load_img(
            'test/pattern/'+k, color_mode='grayscale', target_size=(576,576))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        images.append(input_arr)
        n_id.append(k.strip('.png'))
    return np.asarray(images), n_id