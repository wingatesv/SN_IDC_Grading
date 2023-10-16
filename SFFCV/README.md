# Five-Fold Cross-Validation CSV Generator

This Python script generates five-fold cross-validation CSV files from a given dataset. The dataset should be structured such that each class is in its own directory.

## Requirements
- Google Colab
- Python 3.x
- pandas
- sklearn
- pathlib
- numpy
- tqdm
- argparse

You can install the required Python packages using pip:

`!pip install pandas sklearn pathlib numpy tqdm argparse`


## Usage

You can run the script using the following command:

`!python /content/form_sffcv.py --base_path /content/content/FBCG_Dataset/Training_Set --save_dir /content/`


Replace `/content/form_sffcv.py` with the path to the Python script, `/content/content/FBCG_Dataset/Training_Set` with the base path of your dataset, and `/content/` with the directory where you want to save the CSV files.

### Arguments

- `--base_path`: The base path of the dataset. Each class should be in its own directory under this base path.
- `--save_dir`: The directory to save the CSV files.

## Output

The script will generate five pairs of training and validation CSV files in the specified save directory. Each pair corresponds to a fold in the five-fold cross-validation. The CSV files are named `training_data_400x{i}.csv` and `validation_data_400


# Script: run_sffcv_and_final_train_test.py

This script performs cross-validation (CV) and final training and testing of machine learning models on a given dataset using various machine learning models. The script covers both CV and final training and testing phases.

## Prerequisites

- Python 3.6+
- Required Python packages (See the "install_required_packages" function in the script)

## Usage

1. Configure the script by modifying the parameters and directories in the "if __name__ == '__main__':" section.

2. Run the script by executing the following command:
   
```bash
python run_sffcv_and_final_train_test.py
```

## This will execute both the cross-validation and final training and testing phases for the specified dataset and machine learning model.

## Script Structure

The script is divided into three parts:

### Part 1: Cross Validation

In this part, the script performs cross-validation by training and testing the specified machine learning model on different folds of the dataset. The following tasks are performed:

- Data loading and preprocessing
- Model creation and compilation
- Training the model
- Plotting loss and accuracy curves
- Evaluating the model on each fold
- Generating ROC AUC curves
- Calculating inference time and recording results

### Part 2: Final Train Test

This part performs final training and testing on the entire dataset using the trained model. The steps include:

- Data loading and preprocessing for the final training and testing
- Model creation and compilation
- Training the model on the entire dataset
- Plotting loss and accuracy curves for the final training
- Evaluating the model on the test dataset
- Generating ROC AUC curves
- Calculating inference time and recording results

### Part 3: Get the final results

This part calculates and prints statistical results based on the cross-validation results. It calculates the mean and standard deviation of various performance metrics, including Balanced Accuracy, Macro Precision, Macro Recall, Macro F1-score, Kappa Score, Inference Time, and Training Time. The results are displayed on the console and saved to a text file.

### Example:

```bash
!python /content/run_sffcv_and_final_train_test.py \
--model_name EB0 \
--dataset_name V_Temp5_FBCG_Dataset \
--image_size 224 \
--batch_size 16 \
--epochs 1 \
--learning_rate 0.001 \
--base_train_csv_dir '/content/training_data' \
--base_val_csv_dir '/content/validation_data' \
--cv_result_dir '/content/V_Temp5_FBCG_Dataset_cv_result.csv' \
--save_figure_dir '/content/figure' \
--output_cv_result_dir '/content/results/cv_results.txt' \
--train_dir '/content/V_Temp5_FBCG_Dataset/Training Set' \
--test_dir '/content/V_Temp5_FBCG_Dataset/Test Set' \
--test_result_dir '/content/results/V_Temp5_FBCG_Dataset_test_result.csv'
```
