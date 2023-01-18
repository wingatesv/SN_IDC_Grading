# -*- coding: utf-8 -*-
"""
# Install Package
"""

!git clone https://github.com/khtao/StainGAN.git
!git clone https://github.com/khtao/StainNet.git
!pip install dominate
!pip install visdom

"""# Import Libraries"""

import itertools
import os
import statistics
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import PIL
from google.colab import drive
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle 
import sklearn
import sklearn.model_selection
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from keras import backend as K
from sklearn.utils import class_weight 
import pathlib
import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc,  roc_curve, balanced_accuracy_score, cohen_kappa_score
from scipy import interp
import time
import sys
import cv2 as cv

import torch
from StainNet.models import StainNet, ResnetGenerator

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

"""# Mount Drive"""

drive.mount('/content/drive')

"""# Initialise Functions"""

def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
  """
  Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.
  Some examples of different formats of class_series and their outputs are:
    - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
    - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
    {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
    - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
    - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
    {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}
  The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
  of appareance of the label when the dataset was processed. 
  In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).
  Author: Angel Igareta (angel@igareta.com)
  """
  if multi_class:
    # If class is one hot encoded, transform to categorical labels to use compute_class_weight   
    if one_hot_encoded:
      class_series = np.argmax(class_series, axis=1)
  
    # Compute class weights with sklearn method
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    return dict(zip(class_labels, class_weights))
  else:
    # It is neccessary that the multi-label values are one-hot encoded
    mlb = None
    if not one_hot_encoded:
      mlb = MultiLabelBinarizer()
      class_series = mlb.fit_transform(class_series)

    n_samples = len(class_series)
    n_classes = len(class_series[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1
    
    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))



def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def create_model(model, class_weights):

  model_handle_map = {
  "EB0V2-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
  "EB0V2": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
  "EB0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
  "RV1": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5",
  "RV2": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
  "MB1": "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5",
  "MB2": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
  }
  
  model_handle = model_handle_map.get(model)
  print(f"Selected model: {model} : {model_handle}")
  print("Building model with", model_handle)
  do_fine_tuning = False

  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),                                       
  ])


  base_model = hub.KerasLayer(model_handle, trainable=do_fine_tuning)
  dense_layer = tf.keras.layers.Dense(256, activation='relu')
  prediction_layer = tf.keras.layers.Dense(4, activation='softmax')

  inputs = tf.keras.Input(shape=IMAGE_SIZE+(3,))
  x = data_augmentation(inputs)
  x = base_model(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = dense_layer(x)
  x = tf.keras.layers.Dropout(0.4)(x) 
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs, name = model)

  weights = np.fromiter(class_weights.values(), dtype=float)
  print('Class weights:',weights)
  loss = weighted_categorical_crossentropy(weights)

  model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
  loss=loss,
  metrics=['accuracy'])

  model.summary()
  
  return model




def plot_confusion_matrix(cm, classes, dir,
                          normalize=False,
                          title='Confusion matrix',
                          
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(dir)
    plt.close()

"""# Setup SN Dataset"""

# Reinhard
sn_label = 'SG'


# original dataset
base_path = "/content/drive/Shareddrives/Unlimited Drive/SN/FBCG Dataset"



#  make a temp directory to save SN dataset
save_path = f'/content/{sn_label}'
if not os.path.exists(save_path):
  os.mkdir(save_path)

"""# Setup Model"""

# Set hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
print('Batch size:', BATCH_SIZE, 'Epoch: ', EPOCHS)

best_weight_dir = f'/content/drive/Shareddrives/Unlimited Drive/SN/Weight/{sn_label}'
if not os.path.exists(best_weight_dir):
   os.mkdir(best_weight_dir)

   
figure_dir = '/content/drive/Shareddrives/Unlimited Drive/SN/Result/Figures'




metrics = {'Model Name': [], 'Balanced Accuracy': [], 'Macro Precision': [], 'Macro Recall': [], 'Macro F1score': [], 'Kappa Score': []}
result_dir = f'/content/drive/Shareddrives/Unlimited Drive/SN/Result/{sn_label}_result.csv'
if not os.path.exists(result_dir):
  df = pd.DataFrame(metrics) 
  df.to_csv(result_dir, index=False)

"""# Form SN datasets"""

def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image=image[np.newaxis, ...]
    image=torch.from_numpy(image)
    return image

def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1,2,0))
    return image

#load  pretrained StainGAN
model_GAN = ResnetGenerator(3, 3, ngf=64, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9).cuda().cuda()
model_GAN.load_state_dict(torch.load("/content/StainNet/checkpoints/camelyon16_dataset/latest_net_G_A.pth"))

print(f'Folder contains {len(os.listdir(base_path))} folders')
print(os.listdir(base_path))

print('')

for type_folder in os.listdir(base_path):
  type_path = os.path.join(base_path, type_folder)
  print(f'In Folder :{type_path}')
  print('')

  for grade_folder in os.listdir(type_path):
    grade_path = os.path.join(type_path, grade_folder)
    print(f'In Folder :{grade_path}')
    print('')

    for files in os.listdir(grade_path):
      
      image_path = os.path.join(grade_path, files)
      img_source = PIL.Image.open(image_path)
      image_gan=model_GAN(norm(img_source).cuda())
      image_gan=un_norm(image_gan)


      saveim_path = os.path.join(save_path, type_folder, grade_folder)
      if not os.path.exists(saveim_path):
        os.makedirs(saveim_path)

        
      os.chdir(saveim_path) 
      file_name = str(files)
      cv.imwrite(file_name,image_gan)

"""# Train and Test Models"""

model_list = ['EB0', 'EB0V2', 'EB0V2-21k', 'RV1', 'RV2', 'MB1', 'MB2']
# train_dir =f'/content/{sn_label}/Training Set'
# test_dir =f'/content/{sn_label}/Test Set'

train_dir = "/content/drive/Shareddrives/Unlimited Drive/SN/SG/Training Set"
test_dir = "/content/drive/Shareddrives/Unlimited Drive/SN/SG/Test Set"


train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  seed=123,
  image_size= IMAGE_SIZE,
  batch_size=BATCH_SIZE,
  label_mode='categorical')

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=IMAGE_SIZE,
  batch_size=BATCH_SIZE,
  label_mode = 'categorical')

class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.Rescaling(1. / 255)

train_ds = train_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))
test_ds = test_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))

y = np.concatenate([y for x, y in train_ds], axis=0)
class_weights = generate_class_weights(y, multi_class=True, one_hot_encoded=True)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


for model_name in model_list:
  model = create_model(model_name, class_weights)


  best_weight_path = f'{best_weight_dir}/{model_name}_best.h5'
  cp_callback = tf.keras.callbacks.ModelCheckpoint(best_weight_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') 


  history=model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=EPOCHS,
  callbacks = [cp_callback], 
  class_weight = class_weights
  )

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
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.savefig(f'{figure_dir}/{sn_label}_{model_name}_Train_Val_Curve.png')
  plt.show()
  plt.close()

  model.load_weights(best_weight_path)

  loss, accuracy = model.evaluate(test_ds)
  print('Test accuracy :', accuracy)

  test_labels = np.concatenate([y for x, y in test_ds], axis=0)
  predictions = model.predict(test_ds)
  y_pred = np.argmax(predictions, axis=1)
  y_label = np.argmax(test_labels, axis=1)

  cm = confusion_matrix(y_label, y_pred)
  plot_confusion_matrix(cm, classes= class_names, dir = f'{figure_dir}/{sn_label}_{model_name}_CM.png' )
  plot_confusion_matrix(cm, normalize = True, classes = class_names, dir = f'{figure_dir}/{sn_label}_{model_name}_CM_Norm.png')


  print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_label, y_pred)))
  print('\nBalanced Accuracy: {:.4f}\n'.format(balanced_accuracy_score(y_label, y_pred)))

  print('Macro Precision: {:.4f}'.format(precision_score(y_label, y_pred, average='macro')))
  print('Macro Recall: {:.4f}'.format(recall_score(y_label, y_pred, average='macro')))
  print('Macro F1-score: {:.4f}\n'.format(f1_score(y_label, y_pred, average='macro')))
  print('Kappa score: {:.4f}'.format(cohen_kappa_score(y_label, y_pred)))

  print('\nClassification Report\n')
  print(classification_report(y_label, y_pred, target_names=class_names))

  baccuracy = round(balanced_accuracy_score(y_label, y_pred), 4)
  macro_precision = round(precision_score(y_label, y_pred, average='macro'), 4)
  macro_recall = round(recall_score(y_label, y_pred, average='macro'), 4)
  macro_f1score = round(f1_score(y_label, y_pred, average='macro'), 4)
  kappa_score = round(cohen_kappa_score(y_label, y_pred), 4)

  y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=4)
  y_label = tf.keras.utils.to_categorical(y_label, num_classes=4)

  # Plot ROC Curve
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  n_classes = 4
  for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_label[:,i], y_pred[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])


  plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Grade 0 vs Rest, (area = %0.2f) ' % roc_auc[0])
  plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Grade 1 vs Rest, (area = %0.2f) ' % roc_auc[1])
  plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Grade 2 vs Rest, (area = %0.2f) ' % roc_auc[2])
  plt.plot(fpr[3], tpr[3], linestyle='--',color='red', label='Grade 3 vs Rest, (area = %0.2f) ' % roc_auc[3])
  plt.title('Receiver operating characteristic for IDC Grade 0-3')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive rate')
  plt.legend(loc='best')
  plt.savefig(f'{figure_dir}/{sn_label}_{model_name}_ROC.png')
  plt.show()
  plt.close()



  df = pd.DataFrame({'Model Name': model_name, 'Balanced Accuracy': baccuracy, 'Macro Precision': macro_precision, 'Macro Recall': macro_recall, 'Macro F1score': macro_f1score, 'Kappa Score': kappa_score}, index=[0])  

  df.to_csv(result_dir,  mode='a', index=False, header=False)
