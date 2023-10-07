# import libraries
import pandas as pd
from sklearn.utils import shuffle 
import sklearn
import sklearn.model_selection
from pathlib import Path
import os
import numpy as np
import argparse
from tqdm import tqdm

def main(base_path, save_dir):
    # get image file path
    path = Path(base_path)
    print('Base Path: ', path)
    file_path = [str(p) for p in path.glob("**/*") if p.is_file()]
    print(f"Total number of images: {len(file_path)}")

    # create a df based on file path and class
    pd.set_option("display.max_colwidth", 0)
    data = pd.DataFrame(index=np.arange(0, len(file_path)), columns=["path", "grade"])

    for count, image_path in enumerate(tqdm(file_path, desc="Processing images")):
        grade = os.path.basename(os.path.dirname(image_path)) # Extract grade from path

        data.at[count, "path"] = image_path
        data.at[count, "grade"] = grade

    print(f"Grades: {data.grade.unique()}")
    print(f"Data shape: {data.shape}")

    # shuffle df
    data = shuffle(data, random_state=123)

    # get the label
    Y = data[['grade']]

    # initialise stratifed kfold
    skf = sklearn.model_selection.StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

    # generate each fold and save as csv
    i = 1
    for train_index, val_index in skf.split(np.zeros(len(data)),Y):
        training_data = data.iloc[train_index]
        validation_data = data.iloc[val_index]
        training_data.to_csv(f'{save_dir}/training_data_400x{i}.csv')
        validation_data.to_csv(f'{save_dir}/validation_data_400x{i}.csv')
        print(f"Fold {i} CSV files have been saved.")
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate five-fold cross-validation CSV files.')
    parser.add_argument('--base_path', type=str, help='The base path of the dataset.')
    parser.add_argument('--save_dir', type=str, help='The directory to save the CSV files.')
    
    args = parser.parse_args()
    
    main(args.base_path, args.save_dir)

