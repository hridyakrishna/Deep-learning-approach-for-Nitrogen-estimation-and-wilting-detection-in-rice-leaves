Original file is located at
    https://colab.research.google.com/drive/1RuBaUDS9nsotpmeMB2mx3C1ShBd17JMd
"""
#IMPORTING NECESSARY LIBRARIES

import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2
# %matplotlib inline
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

#MOUNTING GOOGLE DRIVE

from google.colab import drive
drive.mount("/content/gdrive")

#SETTING PATH FOR DATASET

import os
import matplotlib.pyplot as plt
os.chdir("/content/gdrive/MyDrive/NitrogenEstimationTest")
input_path = "/content/gdrive/MyDrive/NitrogenEstimationTest/"

#PLOTTING FEW IMAGES OF EACH CLASS

fig, ax = plt.subplots(8, 2, figsize=(10, 10))
ax = ax.ravel()
plt.tight_layout()
for i, _set in enumerate(['Test', 'Train']):
  set_path = input_path + _set
  ax[i].imshow(plt.imread(set_path+'/swap1_normal/'+os.listdir(set_path+'/swap1_normal')[0]), cmap='gray')
  ax[i].set_title('Set: {}, Condition: swap1_normal'.format(_set))
  ax[i+2].imshow(plt.imread(set_path+'/swap2_normal/'+os.listdir(set_path+'/swap2_normal')[0]), cmap='gray')
  ax[i+2].set_title('Set: {}, Condition: swap2_normal'.format(_set))
  ax[i+4].imshow(plt.imread(set_path+'/swap3_normal/'+os.listdir(set_path+'/swap3_normal')[0]), cmap='gray')
  ax[i+4].set_title('Set: {}, Condition: swap3_normal'.format(_set))
  ax[i+6].imshow(plt.imread(set_path+'/swap4_normal/'+os.listdir(set_path+'/swap4_normal')[0]), cmap='gray')
  ax[i+6].set_title('Set: {}, Condition: swap4_normal'.format(_set))

  ax[i+8].imshow(plt.imread(set_path+'/swap1_wilted/'+os.listdir(set_path+'/swap1_wilted')[0]), cmap='gray')
  ax[i+8].set_title('Set: {}, Condition: swap1_wilted'.format(_set))
  ax[i+10].imshow(plt.imread(set_path+'/swap2_wilted/'+os.listdir(set_path+'/swap2_wilted')[0]), cmap='gray')
  ax[i+10].set_title('Set: {}, Condition: swap2_wilted'.format(_set))
  ax[i+12].imshow(plt.imread(set_path+'/swap3_wilted/'+os.listdir(set_path+'/swap3_wilted')[0]), cmap='gray')
  ax[i+12].set_title('Set: {}, Condition: swap3_normal'.format(_set))
  ax[i+14].imshow(plt.imread(set_path+'/swap4_wilted/'+os.listdir(set_path+'/swap4_wilted')[0]), cmap='gray')
  ax[i+14].set_title('Set: {}, Condition: swap4_wilted'.format(_set))

#PRINTING SIZE OF IMAGES IN EACH CLASS 

for _set in ['Test','Train']:
    n_swap1_normal = len(os.listdir(input_path + _set + '/swap1_normal'))
    n_swap2_normal = len(os.listdir(input_path + _set + '/swap2_normal'))
    n_swap3_normal = len(os.listdir(input_path + _set + '/swap3_normal'))
    n_swap4_normal = len(os.listdir(input_path + _set + '/swap4_normal'))
    n_swap1_wilted = len(os.listdir(input_path + _set + '/swap1_wilted'))
    n_swap2_wilted = len(os.listdir(input_path + _set + '/swap2_wilted'))
    n_swap3_wilted = len(os.listdir(input_path + _set + '/swap3_wilted'))
    n_swap4_wilted = len(os.listdir(input_path + _set + '/swap4_wilted'))
    print('Set: {}, swap1_normal images: {}, swap1_wilted images: {}, swap2_normal images: {}, swap2_wilted images: {}, swap3_normal images: {}, swap3_wilted images: {}, swap4_normal images: {}, swap4_wilted images: {},'.format(_set, n_swap1_normal, n_swap1_wilted, n_swap2_normal, n_swap2_wilted, n_swap3_normal, n_swap3_wilted, n_swap4_normal, n_swap4_wilted))

#PREPROCESSING THE DATASET (COLOUR CONVERSION AND NOISE REMOVING USING GAUSSIAN SMOOTHING)

def func(path):    
    frame = cv2.imread(path)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

#SETTING PATH OF TRAIN AND TEST SET

trainpath = "/content/gdrive/MyDrive/NitrogenEstimationTest/Train/"
testpath = "/content/gdrive/MyDrive/NitrogenEstimationTest/Test/"

#IMPORTING NECESSARY LIBRARIES

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil

#FEATURE EXTRACTION USING INCEPTIONV3

def image_feature(directory):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    for j in os.listdir(path):
      spec_path=path+j
      for i in os.listdir(spec_path):
          fname = path+j+'/'+i
          img=image.load_img(fname,target_size=(224,224))
          x = img_to_array(img)
          x=np.expand_dims(x,axis=0)
          x=preprocess_input(x)
          feat=model.predict(x)
          feat=feat.flatten()
          features.append(feat)
          img_name.append(i)
    return features,img_name

#IMAGE SEGMENTATION USING K-MEANS

path = "/content/gdrive/MyDrive/NitrogenEstimationTest/Train/"
img_path=os.listdir(path)
img_features,img_name=image_feature(img_path)

k = 8
clusters = KMeans(k, random_state = 40)
clusters.fit(img_features)

image_cluster = pd.DataFrame(img_name,columns=['image'])
image_cluster["clusterid"] = clusters.labels_
image_cluster

# DATA AUGMENTATION

train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
train_generator = train_datagen.flow_from_directory(directory=trainpath,target_size=(128,128),class_mode='categorical',batch_size=1)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=testpath,target_size=(128,128),class_mode='categorical',batch_size=1)

train_generator.class_indices

#IMPORTING NECESSARY LIBRARIES

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense

#IMAGE CLASSIFICATION USING VGG-19

vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))

for layer in vgg19.layers:
    layer.trainable = False
    
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(8,activation='softmax'))
model.summary()

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=10,
                              epochs=10,validation_data=test_generator,
                              validation_steps=None)

model.evaluate_generator(test_generator)

#PLOTTING THE OBSERVATIONS

import matplotlib.pyplot as plt
acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, test_acc, 'b', label='Testing acc')
plt.title('Training and Testing accuracy')
plt.legend()
plt.figure()

loss = history.history['loss']
test_loss = history.history['val_loss']
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, test_loss, 'b', label='Testing loss')
plt.title('Training and Testing loss')
plt.legend()
plt.figure()

#METRICS ANALYSIS

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
Y_pred = model.predict_generator(test_generator,186)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['swap1_normal','swap1_wilted','swap2_normal','swap2_wilted','swap3_normal','swap3_wilted','swap4_normal','swap4_wilted']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
