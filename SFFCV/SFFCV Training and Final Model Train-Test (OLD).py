
"""   
This is a sample code for model SFFCV training and retraining-testing with EfficientNetB0
The same code is applicable for other models
"""

import itertools
import os
import statistics
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import PIL
import PIL.Image
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

# Select Pre-trained CNN
model_name = "efficientnet_b0"
model_handle_map = {
  "efficientnetv2-b0-21k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
  "efficientnetv2-b0": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
  "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
  "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/4",
  "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/4",
  "mobilenet_v1_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5",
  "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
}


# Setup Hyperparameters
model_handle = model_handle_map.get(model_name)
print(f"Selected model: {model_name} : {model_handle}")

IMAGE_SIZE = (224, 224)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = 16
EPOCHS = 100
print('Batch size: ', BATCH_SIZE, 'Epoch: ', EPOCHS)


# directory to save
save_dir = '/content/'
save_model = f"/content/V2_{model_name}_CrossValidation400x_"
checkpoint_path = f'/content/v2_cp_{model_name}_400x_CV.h5'


#Create csv to save results
metrics = {'Balanced Accuracy': [], 'Macro Precision': [], 'Macro Recall': [], 'Macro F1score': [], 'Kappa Score': [], 'Inference Time': [], 'Training Time': []}
df = pd.DataFrame(metrics) 
df.to_csv(f'/content/{model_name}_cv_result.csv', index=False)

metrics = {'Balanced Accuracy': [], 'Macro Precision': [], 'Macro Recall': [], 'Macro F1score': [], 'Kappa Score': []}
df = pd.DataFrame(metrics) 
df.to_csv('/content/test_result.csv', index=False)

# Functions
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

def get_model_name(k):
    return 'V2_'+model_name+'_CrossValidation400x_'+str(k)+'.h5'

def plot_confusion_matrix(cm, classes,
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
    plt.close()


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def create_model(class_weights):
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
  model = tf.keras.Model(inputs, outputs, name = model_name)

  weights = np.fromiter(class_weights.values(), dtype=float)
  print(weights)
  loss = weighted_categorical_crossentropy(weights)

  model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
  loss=loss,
  metrics=['accuracy'])

  model.summary()
  
  return model

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




# Cross Validation 
for idx in range(5):
	fold_var = 1 + idx

	training_data = pd.read_csv(f'/content/training_data_400x{fold_var}.csv')
	validation_data = pd.read_csv(f'/content/validation_data_400x{fold_var}.csv')



	train_data_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(training_data, x_col = "path", y_col = "grade", class_mode = "categorical", shuffle = True, seed = 123, batch_size = BATCH_SIZE, target_size= IMAGE_SIZE)
	valid_data_generator  = ImageDataGenerator(rescale=1./255).flow_from_dataframe(validation_data, x_col = "path", y_col = "grade", class_mode = "categorical", shuffle = True, seed = 123, batch_size = BATCH_SIZE, target_size= IMAGE_SIZE)



	class_weights = class_weight.compute_class_weight(class_weight='balanced',classes = np.unique(training_data['grade']),y = training_data['grade'])
	class_weights = dict(enumerate(class_weights))
	print(class_weights)


	model = create_model(class_weights)

	checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
								monitor='val_accuracy', verbose=1, 
								save_best_only=True, mode='max')


	time_callback = TimeHistory()

	callbacks_list = [checkpoint, time_callback]

	history = model.fit(train_data_generator,
				    epochs=EPOCHS,
				    callbacks=callbacks_list,
						class_weight = class_weights,
				    validation_data=valid_data_generator)


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
	plt.show()

	total_training_times = time_callback.times
	summ = np.sum(total_training_times)
	training_time = summ/3600
	print('Total training time: ', training_time, 'hour')

	model.load_weights(save_model+str(fold_var)+".h5")
	print(model_name, " loaded")

	model.evaluate(valid_data_generator)
	nb_samples = valid_data_generator.n

	steps = int(np.ceil(nb_samples/BATCH_SIZE))
	x, y = [], []
	for i in range(steps):
	    a , b = valid_data_generator.next()
	    y.extend(b)
	    x.extend(a)

	x = np.array(x)
	y = np.array(y)
	predict = model.predict(x, steps = np.ceil(nb_samples/BATCH_SIZE))

	y_pred = np.argmax(predict, axis=1)
	y_label = np.argmax(y, axis=1)

	cm = confusion_matrix(y_label, y_pred)
	print(cm)

	print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_label, y_pred)))
	print('\nBalanced Accuracy: {:.4f}\n'.format(balanced_accuracy_score(y_label, y_pred)))

	print('Micro Precision: {:.4f}'.format(precision_score(y_label, y_pred, average='micro')))
	print('Micro Recall: {:.4f}'.format(recall_score(y_label, y_pred, average='micro')))
	print('Micro F1-score: {:.4f}\n'.format(f1_score(y_label, y_pred, average='micro')))

	print('Macro Precision: {:.4f}'.format(precision_score(y_label, y_pred, average='macro')))
	print('Macro Recall: {:.4f}'.format(recall_score(y_label, y_pred, average='macro')))
	print('Macro F1-score: {:.4f}\n'.format(f1_score(y_label, y_pred, average='macro')))
	print('Kappa score: {:.4f}'.format(cohen_kappa_score(y_label, y_pred)))

	print('Weighted Precision: {:.4f}'.format(precision_score(y_label, y_pred, average='weighted')))
	print('Weighted Recall: {:.4f}'.format(recall_score(y_label, y_pred, average='weighted')))
	print('Weighted F1-score: {:.4f}'.format(f1_score(y_label, y_pred, average='weighted')))



	print('\nClassification Report\n')
	print(classification_report(y_label, y_pred, target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']))

	baccuracy = balanced_accuracy_score(y_label, y_pred)
	macro_precision = precision_score(y_label, y_pred, average='macro')
	macro_recall = recall_score(y_label, y_pred, average='macro')
	macro_f1score = f1_score(y_label, y_pred, average='macro')
	kappa_score = cohen_kappa_score(y_label, y_pred)


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
	plt.title('Receiver operating characteristic for Breast Cancer Grade 0-3')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive rate')
	plt.legend(loc='best')

	pred_time_list=[]
	for i in range(10):
	    start = time.process_time() 
	    result = model.predict(x, steps = np.ceil(nb_samples/BATCH_SIZE))
	    end = time.process_time()
	    pred_time_list.append((end-start)/183)
	inference_time = sum(pred_time_list)/10
	print('Inference time: ',inference_time, 'second') 

	df = pd.DataFrame({'Balanced Accuracy': baccuracy, 'Macro Precision': macro_precision, 'Macro Recall': macro_recall, 'Macro F1score': macro_f1score, 'Kappa Score': kappa_score, 'Inference Time': inference_time, 'Training Time': training_time}, index=[0])  
	df.to_csv(f'/content/drive/MyDrive/Wingates FYP/CSV2/{model_name}_result.csv',  mode='a', index=False, header=False)
	
	
	
	
# Retrain model with whole train set and test model with test set
train_dir ='/content/FBCG Dataset/Training Set'
test_dir ='/content/FBCG Dataset/Test Set'

train_dir = pathlib.Path(train_dir)
test_dir = pathlib.Path(test_dir)


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

model = create_model(class_weights)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') 

time_callback = TimeHistory()


history=model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=EPOCHS,
  callbacks = [cp_callback, time_callback], 
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
plt.show()



total_training_times = time_callback.times
summ = np.sum(total_training_times)
print('Total training time: ', summ/3600, 'hour')

model.load_weights(checkpoint_path)
print(model_name, " loaded")
loss, accuracy = model.evaluate(test_ds)
print('Test accuracy :', accuracy)

test_labels = np.concatenate([y for x, y in test_ds], axis=0)
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)
y_label = np.argmax(test_labels, axis=1)

cm = confusion_matrix(y_label, y_pred)

print('\nAccuracy: {:.4f}\n'.format(accuracy_score(y_label, y_pred)))
print('\nBalanced Accuracy: {:.4f}\n'.format(balanced_accuracy_score(y_label, y_pred)))

print('Micro Precision: {:.4f}'.format(precision_score(y_label, y_pred, average='micro')))
print('Micro Recall: {:.4f}'.format(recall_score(y_label, y_pred, average='micro')))
print('Micro F1-score: {:.4f}\n'.format(f1_score(y_label, y_pred, average='micro')))

print('Macro Precision: {:.4f}'.format(precision_score(y_label, y_pred, average='macro')))
print('Macro Recall: {:.4f}'.format(recall_score(y_label, y_pred, average='macro')))
print('Macro F1-score: {:.4f}\n'.format(f1_score(y_label, y_pred, average='macro')))
print('Kappa score: {:.4f}'.format(cohen_kappa_score(y_label, y_pred)))

print('Weighted Precision: {:.4f}'.format(precision_score(y_label, y_pred, average='weighted')))
print('Weighted Recall: {:.4f}'.format(recall_score(y_label, y_pred, average='weighted')))
print('Weighted F1-score: {:.4f}'.format(f1_score(y_label, y_pred, average='weighted')))


print('\nClassification Report\n')
print(classification_report(y_label, y_pred, target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']))

baccuracy = balanced_accuracy_score(y_label, y_pred)
macro_precision = precision_score(y_label, y_pred, average='macro')
macro_recall = recall_score(y_label, y_pred, average='macro')
macro_f1score = f1_score(y_label, y_pred, average='macro')
kappa_score = cohen_kappa_score(y_label, y_pred)

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
plt.title('Receiver operating characteristic for Breast Cancer Grade 0-3')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')

pred_time_list=[]
for i in range(10):
    start = time.process_time() 
    result = model.predict(test_ds)
    end = time.process_time()
    pred_time_list.append((end-start)/183)
print('Inference time: ',sum(pred_time_list)/10, 'second') 




df = pd.DataFrame({'Balanced Accuracy': baccuracy, 'Macro Precision': macro_precision, 'Macro Recall': macro_recall, 'Macro F1score': macro_f1score, 'Kappa Score': kappa_score}, index=[0])  
df.to_csv('/content/test_result.csv',  mode='a', index=False, header=False)


# Compute cv results 
def Average(lst):
    return sum(lst)/ len(lst)

def calculate(x):
  average = Average(x)
  print('{:.4f}'.format(average),'±','{:.4f}'.format(statistics.stdev(x)))

df = pd.read_csv(f'/content/{model_name}_result.csv')
print(df.to_string()) 

col_acc_list = df['Balanced Accuracy'].tolist()
col_pre_list = df['Macro Precision'].tolist()
col_rec_list = df['Macro Recall'].tolist()
col_f1_list = df['Macro F1score'].tolist()
col_kappa_list = df['Kappa Score'].tolist()
col_it_list = df['Inference Time'].tolist()
col_tt_list = df['Training Time'].tolist()

calculate(col_acc_list)
calculate(col_pre_list)
calculate(col_rec_list)
calculate(col_f1_list)
calculate(col_kappa_list)
calculate(col_it_list)
calculate(col_tt_list)
