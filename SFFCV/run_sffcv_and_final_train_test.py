import subprocess
import pkg_resources

def install_required_packages():
    # List of required packages
    required_packages = ['keras-preprocessing']

    installed_packages = [pkg.key for pkg in pkg_resources.working_set]

    missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

    if missing_packages:
        try:
            for package in missing_packages:
                # Install missing packages
                subprocess.run(["pip", "install", package])

            print("Packages", ", ".join(missing_packages), "have been successfully installed.")
        except Exception as e:
            print("An error occurred during package installation:", str(e))
    else:
        print("Packages 'keras-preprocessing' is already installed.")

install_required_packages()
import os
import time
import pathlib
import argparse
import statistics
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, auc, balanced_accuracy_score,
    cohen_kappa_score, roc_curve
)
from scipy import interp
import matplotlib.pylab as plt

# Define weighted categorical cross-entropy loss function
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

# Function to get the checkpoint cv model path
def get_cv_ckpt_model_path(k, dataset, model_name):
    return f'/content/{model_name}_{dataset}_cv'+str(k)+'.h5'

# Function to get the checkpoint model path
def get_ckpt_model_path(dataset, model_name):
    return f'/content/{model_name}_{dataset}.h5'

# Callback for tracking training time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Create and compile a machine learning model
def create_and_compile_model(model, class_weights, image_size, learning_rate):

  """
  Create and compile a machine learning model.

  Args:
      model (str): Name of the model.
      class_weights (dict): Class weights for balancing.
      image_size (tuple): Target image size.

  Returns:
      model: Compiled machine learning model.
  """

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

  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
  ])


  base_model = hub.KerasLayer(model_handle, trainable=False)
  dense_layer = tf.keras.layers.Dense(256, activation='relu')
  prediction_layer = tf.keras.layers.Dense(4, activation='softmax')

  inputs = tf.keras.Input(shape=image_size+(3,))
  x = data_augmentation(inputs)
  x = base_model(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  x = dense_layer(x)
  x = tf.keras.layers.Dropout(0.4)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs, name = model)

  weights = np.fromiter(class_weights.values(), dtype=float)

  model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
  loss=weighted_categorical_crossentropy(weights),
  metrics=['accuracy'])


  return model

# Generate class weights based on class labels
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

def cv_load_and_create_data_generators(base_train_csv_dir, base_val_csv_dir, fold_var, batch_size, image_size):
    """
    Load training and validation data for a specific fold and create data generators.

    Args:
        fold_var (int): Fold number.
        batch_size (int): Batch size.
        image_size (tuple): Target image size.

    Returns:
        train_data_generator: Training data generator.
        valid_data_generator: Validation data generator.
    """
    print('Reading ', f'{base_train_csv_dir}_{fold_var}.csv')
    training_data = pd.read_csv(f'{base_train_csv_dir}_{fold_var}.csv')
    print('Reading ', f'{base_val_csv_dir}_{fold_var}.csv')
    validation_data = pd.read_csv(f'{base_val_csv_dir}_{fold_var}.csv')

    train_data_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        training_data, x_col="path", y_col="grade", class_mode="categorical",
        shuffle=True, seed=123, batch_size=batch_size, target_size=image_size
    )

    valid_data_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        validation_data, x_col="path", y_col="grade", class_mode="categorical",
        shuffle=True, seed=123, batch_size=batch_size, target_size=image_size
    )

    class_weights = dict(enumerate(class_weight.compute_class_weight(class_weight='balanced',classes = np.unique(training_data['grade']),y = training_data['grade'])))

    return train_data_generator, valid_data_generator, class_weights

def train_model(model, train_set, val_set, class_weights, epochs, dataset_name, model_name, cv=True, fold_var = None):
    """
    Train the machine learning model.

    Args:
        model: Compiled machine learning model.
        train_data_generator: Training data generator.
        valid_data_generator: Validation data generator.
        epochs (int): Number of training epochs.
        fold_var (int): Fold number.
        dataset_name (str): Name of the dataset.

    Returns:
        history: Training history.
        training_time (float): Total training time in hours.
    """
    if cv:
      ckpt_path = get_cv_ckpt_model_path(fold_var, dataset_name, model_name)
    else:
      ckpt_path = get_ckpt_model_path(dataset_name, model_name)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                  monitor='val_accuracy', verbose=1,
                  save_best_only=True, mode='max')


    time_callback = TimeHistory()

    callbacks_list = [checkpoint, time_callback]

    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=epochs,
        callbacks=callbacks_list,
        class_weight=class_weights,
       
    )


    training_time = round(np.sum(time_callback.times)/3600,4)
    print('Total training time: ', training_time, 'hour')

    return history, training_time

def plot_loss_acc_curve(history, save_figure_dir, dataset_name, model_name, cv=True, fold_var=None):
  """
  Plot loss and accuracy curves.

  Args:
      history: Training history.
  """
  #Create the directory if it doesn't exist
  if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
      
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
  if cv:
    filename = f'{dataset_name}_{model_name}_k{fold_var}_train_val_graph.png'
  else:
    filename = f'{dataset_name}_{model_name}_train_val_graph.png'
  plt.savefig(os.path.join(save_figure_dir, filename))
  plt.close()
  print('Created Figure: ', os.path.join(save_figure_dir, filename))


def roc_auc_curve(y_label, y_pred, save_figure_dir, dataset_name, model_name, cv=True, fold_var=None):
  #Create the directory if it doesn't exist
  if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)
      
  y_pred = tf.keras.utils.to_categorical(y_pred, num_classes=4)
  y_label = tf.keras.utils.to_categorical(y_label, num_classes=4)
  # Plot ROC Curve
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(4):
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
  if cv:
    filename = f'{dataset_name}_{model_name}_k{fold_var}_roc_auc.png'
  else:
    filename = f'{dataset_name}_{model_name}_roc_auc.png'
  plt.savefig(os.path.join(save_figure_dir, filename))
  plt.close()
  print('Created Figure: ', os.path.join(save_figure_dir, filename))

def generate_inference_time(model, x, nb_samples, batch_size, cv=True):
  pred_time_list=[]
  for i in range(10):
    start = time.process_time()
    if cv:
      result = model.predict(x, steps = np.ceil(nb_samples/batch_size), verbose=0)
    else:
      result = model.predict(x, verbose=0)
    end = time.process_time()
    pred_time_list.append((end-start)/nb_samples)
  inference_time = round(sum(pred_time_list)/10,4)
  print('\nInference time: ',inference_time, 'second\n')

  return inference_time

def generate_scores(y_label, y_pred):
  cm = confusion_matrix(y_label, y_pred)
  print('Confusion Matrix\n')
  print(cm)

  print('\nBalanced Accuracy: {:.4f}'.format(balanced_accuracy_score(y_label, y_pred)))
  print('Macro Precision: {:.4f}'.format(precision_score(y_label, y_pred, average='macro')))
  print('Macro Recall: {:.4f}'.format(recall_score(y_label, y_pred, average='macro')))
  print('Macro F1-score: {:.4f}'.format(f1_score(y_label, y_pred, average='macro')))
  print('Kappa score: {:.4f}'.format(cohen_kappa_score(y_label, y_pred)))

  baccuracy = round(balanced_accuracy_score(y_label, y_pred),4)
  macro_precision = round(precision_score(y_label, y_pred, average='macro'),4)
  macro_recall = round(recall_score(y_label, y_pred, average='macro'),4)
  macro_f1score = round(f1_score(y_label, y_pred, average='macro'),4)
  kappa_score = round(cohen_kappa_score(y_label, y_pred),4)

  return  baccuracy, macro_precision, macro_recall, macro_f1score, kappa_score

def cv_evaluate_model(model, valid_data_generator, model_name, fold_var, dataset_name, batch_size, training_time, cv_result_dir, save_figure_dir):
  """
  Evaluate the trained model.

  Args:
      model: Trained machine learning model.
      valid_data_generator: Validation data generator.
      model_name (str): Name of the model.
      fold_var (int): Fold number.
      dataset_name (str): Name of the dataset.
      batch_size (int): Batch size.
      cv_result_dir (str): Directory to store evaluation results.
  """
  model.load_weights(get_cv_ckpt_model_path(fold_var, dataset_name, model_name))

  model.evaluate(valid_data_generator, verbose=0)
  nb_samples = valid_data_generator.n

  steps = int(np.ceil(nb_samples/batch_size))
  x, y = [], []
  for i in range(steps):
      a , b = valid_data_generator.next()
      y.extend(b)
      x.extend(a)

  x = np.array(x)
  y = np.array(y)
  predict = model.predict(x, steps = np.ceil(nb_samples/batch_size), verbose=0)

  y_pred = np.argmax(predict, axis=1)
  y_label = np.argmax(y, axis=1)
  

  baccuracy, macro_precision, macro_recall, macro_f1score, kappa_score = generate_scores(y_label, y_pred)

  roc_auc_curve(y_label, y_pred, save_figure_dir, dataset_name, model_name, cv=True, fold_var=fold_var)

  inference_time = generate_inference_time(model, x, nb_samples, batch_size)

  df = pd.DataFrame({'Model': model_name,'Fold':fold_var,'Balanced Accuracy': baccuracy, 'Macro Precision': macro_precision, 'Macro Recall': macro_recall, 'Macro F1score': macro_f1score, 'Kappa Score': kappa_score, 'Inference Time': inference_time, 'Training Time': training_time}, index=[0])
  df.to_csv(cv_result_dir,  mode='a', index=False, header=False)

def generate_cv_result_csv(cv_result_file):
    # Check if the file exists, and create it if it doesn't
    if not os.path.exists(cv_result_file):
        metrics = {'Model': [], 'Fold': [], 'Balanced Accuracy': [], 'Macro Precision': [],
                   'Macro Recall': [], 'Macro F1score': [], 'Kappa Score': [],
                   'Inference Time': [], 'Training Time': []}
        df = pd.DataFrame(metrics)
        df.to_csv(cv_result_file, index=False)  # Create the CSV file if it doesn't exist
        print(f"Created CSV file: {cv_result_file}")

def generate_test_result_csv(test_result_file):
    # Check if the file exists, and create it if it doesn't
    if not os.path.exists(test_result_file):
        metrics = {'Model': [], 'Balanced Accuracy': [], 'Macro Precision': [],
                   'Macro Recall': [], 'Macro F1score': [], 'Kappa Score': []}
        df = pd.DataFrame(metrics)
        df.to_csv(test_result_file, index=False)  # Create the CSV file if it doesn't exist
        print(f"Created CSV file: {test_result_file}")

def final_train_test_data_preprocessing(train_dir, test_dir, image_size, batch_size):
  print('Reading: ', train_dir)
  print('Reading: ', test_dir)

  train_dir = pathlib.Path(train_dir)
  test_dir = pathlib.Path(test_dir)


  train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size= image_size,
    batch_size=batch_size,
    label_mode='categorical')

  test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode = 'categorical')


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

  return train_ds, test_ds, class_weights

def final_evaluate_model(model, dataset_name, model_name, test_ds, batch_size, save_figure_dir, test_result_dir):
  model.load_weights(get_ckpt_model_path(dataset_name, model_name))
  loss, accuracy = model.evaluate(test_ds, verbose=0)
  print('Final Test accuracy :', round(accuracy,4))

  test_labels = np.concatenate([y for x, y in test_ds], axis=0)
  predictions = model.predict(test_ds, verbose=0)
  y_pred = np.argmax(predictions, axis=1)
  y_label = np.argmax(test_labels, axis=1)


  baccuracy, macro_precision, macro_recall, macro_f1score, kappa_score = generate_scores(y_label, y_pred)

  roc_auc_curve(y_label, y_pred, save_figure_dir, dataset_name, model_name, cv=False)

  num_samples = tf.data.experimental.cardinality(test_ds).numpy()
  inference_time = generate_inference_time(model, test_ds, num_samples, batch_size, cv=False)  



  df = pd.DataFrame({'Model': model_name,'Balanced Accuracy': baccuracy, 'Macro Precision': macro_precision, 'Macro Recall': macro_recall, 'Macro F1score': macro_f1score, 'Kappa Score': kappa_score}, index=[0])
  df.to_csv(test_result_dir,  mode='a', index=False, header=False)


def calculate_and_print_stats(cv_result_dir, output_file):
    def calculate(x, label):
        if len(x) != 5:
            raise ValueError(f"Expected exactly 5 values for {label}, but got {len(x)}")
        mean = sum(x) / len(x)
        stdev = statistics.stdev(x)
        return f'{label}: {mean:.4f} ± {stdev:.4f}'

    df = pd.read_csv(cv_result_dir)
    unique_models = df['Model'].unique()
    result_lines = []

    for model_name in unique_models:
        model_data = df[df['Model'] == model_name]

        if len(model_data) != 5:
            raise ValueError(f"Expected exactly 5 entries for model {model_name}, but got {len(model_data)}")

        columns_to_calculate = {
            'Balanced Accuracy': model_data['Balanced Accuracy'].tolist(),
            'Macro Precision': model_data['Macro Precision'].tolist(),
            'Macro Recall': model_data['Macro Recall'].tolist(),
            'Macro F1score': model_data['Macro F1score'].tolist(),
            'Kappa Score': model_data['Kappa Score'].tolist(),
            'Inference Time': model_data['Inference Time'].tolist(),
            'Training Time': model_data['Training Time'].tolist(),
        }

        model_result_lines = []
        for label, data in columns_to_calculate.items():
            result_line = calculate(data, label)
            model_result_lines.append(result_line)

        # Print to console with model name
        print(f"Model: {model_name}")
        print("\n".join(model_result_lines))

        # Append to the output file with model name
        with open(output_file, 'a') as f:
            f.write(f"Model: {model_name}\n")
            f.write("\n".join(model_result_lines))
            f.write('\n')  # Add a newline to separate multiple runs

def main(args):
    model_name = args.model_name
    dataset_name = args.dataset_name
    image_size = (args.image_size, args.image_size)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    # PART 1: Cross Validation
    print('Cross Validation Phase.............')
    base_train_csv_dir = args.base_train_csv_dir
    base_val_csv_dir = args.base_val_csv_dir
    cv_result_dir = args.cv_result_dir
    save_figure_dir = args.save_figure_dir
    output_cv_result_dir = args.output_cv_result_dir

    # Create the directory if it doesn't exist
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)

    generate_cv_result_csv(cv_result_dir)

    for idx in tqdm(range(5), desc=f"Model: {model_name}"):
        fold_var = 1 + idx
        print(f'\nCross Validation {fold_var} for model {model_name}')
        train_data_generator, valid_data_generator, class_weights = cv_load_and_create_data_generators(base_train_csv_dir, base_val_csv_dir, fold_var, batch_size, image_size)
        model = create_and_compile_model(model_name, class_weights, image_size, learning_rate)
        history, training_time = train_model(model, train_data_generator, valid_data_generator, class_weights, epochs, dataset_name, model_name, cv=True, fold_var=fold_var)
        plot_loss_acc_curve(history, save_figure_dir, dataset_name, model_name, cv=True, fold_var=fold_var)
        cv_evaluate_model(model, valid_data_generator, model_name, fold_var, dataset_name, batch_size, training_time, cv_result_dir, save_figure_dir)

    # PART 2: Final Train Test
    print('Final Train Test Phase.............')
    train_dir = args.train_dir
    test_dir = args.test_dir
    test_result_dir = args.test_result_dir

    generate_test_result_csv(test_result_dir)

    train_ds, test_ds, class_weights = final_train_test_data_preprocessing(train_dir, test_dir, image_size, batch_size)
    model = create_and_compile_model(model_name, class_weights, image_size, learning_rate)
    history, training_time = train_model(model, train_ds, test_ds, class_weights, epochs, dataset_name, model_name, cv=False)
    plot_loss_acc_curve(history, save_figure_dir, dataset_name, model_name, cv=False)
    final_evaluate_model(model, dataset_name, model_name, test_ds, batch_size, save_figure_dir, test_result_dir)

    # PART 3: Get the final results
    calculate_and_print_stats(cv_result_dir, output_cv_result_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument("--model_name", type=str, default='EB0', help="'EB0', 'EB0V2', 'EB0V2-21k', 'RV1', 'RV2', 'MB1', 'MB2'")
    parser.add_argument("--dataset_name", type=str, default='V_Temp5_FBCG_Dataset', help="{R, M, V, ACD}_{Temp1, Temp2, Temp3, Temp4, Temp5, Temp6}_FBCG_Dataset")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (single value, assuming square images)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    # Add arguments for directories and settings
    parser.add_argument("--base_train_csv_dir", type=str, default='/content/training_data', help="Base directory for training CSVs")
    parser.add_argument("--base_val_csv_dir", type=str, default='/content/validation_data', help="Base directory for validation CSVs")
    parser.add_argument("--cv_result_dir", type=str, default='/content/cv_result.csv', help="Directory for CV result CSV")
    parser.add_argument("--save_figure_dir", type=str, default='/content/figure', help="Directory to save figures")
    parser.add_argument("--output_cv_result_dir", type=str, default='/content/cv_results.txt', help="Output CV result directory")
    parser.add_argument("--train_dir", type=str, default='/content/V_Temp5_FBCG_Dataset/Training Set', help="Training data directory")
    parser.add_argument("--test_dir", type=str, default='/content/V_Temp5_FBCG_Dataset/Test Set', help="Test data directory")
    parser.add_argument("--test_result_dir", type=str, default='/content/test_result.csv', help="Directory for test result CSV")

    args = parser.parse_args()
    main(args)

