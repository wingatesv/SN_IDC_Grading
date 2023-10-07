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
