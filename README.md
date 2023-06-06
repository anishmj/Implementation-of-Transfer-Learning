# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for CIFAR-10 dataset classification using VGG-19 architecture.
## Problem Statement and Dataset
To use the pre-trained VGG19 model on the ImageNet dataset and fine-tune it on the CIFAR-10 dataset. The goal is to achieve high accuracy on the CIFAR-10 dataset by leveraging the knowledge learned from the ImageNet dataset. The main challenge is to adapt the pre-trained model to the new dataset while avoiding overfitting and achieving high accuracy

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. the datasetcontains 10 random images of:

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

## DESIGN STEPS
### STEP 1:
Import the required libraries and load the dataset

### STEP 2:
split the dataset for training and testing

### STEP 3:
Set the values of image from 0 t0 1.

### STEP 4:
using the VGG 19 as base model without changing the weights and remove the fully connected layer from VGG19

### STEP 5:
Add our own Fully connected layer to VGG19 base model, compile and fit it


## PROGRAM

```
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from keras import Sequential
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

train_generator = ImageDataGenerator(rotation_range = 2,
                                     horizontal_flip = True,
                                     rescale = 1.0/255.0,
                                     zoom_range = .1)

test_generator = ImageDataGenerator(rotation_range = 2,
                                     horizontal_flip = True,
                                     rescale = 1.0/255.0,
                                     zoom_range = .1)

y_train_onehot = utils.to_categorical(y_train,10)

y_test_onehot = utils.to_categorical(y_test,10)

base_model = VGG19(include_top = False,
                   weights = 'imagenet',
                   input_shape = (32,32,3))

base_model.summary()

single_image = x_train[500]
plt.imshow(single_image,cmap='BrBG')

for layer in base_model.layers:
  layer.trainable = False
y_train.shape
x_train.min()

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

base_model = VGG19(include_top = False,weights = 'imagenet')
for layer in base_model.layers:
  layer.Trainable = False

model = Sequential()
model.add(base_model)



d1 = Dense(units=128,activation = 'relu')
model.add(d1)
drop = Dropout(rate = 0.5)
model.add(drop)
op = Dense(units = 10,activation = 'softmax')
model.add(op)

model.summary()

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

model.fit(x_train_scaled,y_train,epochs =50, batch_size=64,validation_data = (x_test,y_test), callbacks=[learning_rate_reduction])


metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

```


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![p](1.png)

![p](2.png)

### Classification Report

![p](https://user-images.githubusercontent.com/93427246/241361727-16307c30-83c9-4306-bbc9-e29a54240ed9.png)
### Confusion Matrix

![p](https://user-images.githubusercontent.com/93427246/241361712-3dde1f3e-4910-4f62-a2e5-8bef9f8bd581.png)
## RESULT
Thus the implementation of Transfer learning of VGG19 for CIFAR10 dataset is successful.

