
# import libraries
import pandas as pd
from sklearn.utils import shuffle 
import sklearn
import sklearn.model_selection
from fastai import *
from fastai.vision import *

base_path = '/content/FBCG Dataset/Training Set'

# get image file path
path = Path(base_path)
file_path=get_files(path, recurse=True)
print(len(file_path)) # To validate equal total number of images

# create a df based on file path and class
pd.set_option("display.max_colwidth", 0)
data = pd.DataFrame(index=np.arange(0, len(file_path)), columns=["path", "grade"])
count = 0

for image_path in file_path:
  path_name = str(image_path)
  grade = str(path_name[71:78])
  
  
  
  data.iloc[count]["path"] = path_name
  
  data.iloc[count]["grade"] = grade

  count +=1

data.head()
data.grade.unique()
data.shape

# shuffle df
data = shuffle(data)
data.head()

# get the label
Y = data[['grade']]
Y

# initialise stratifed kfold
skf = sklearn.model_selection.StratifiedKFold (n_splits = 5, random_state = 123, shuffle = True)

# generate each fold and save as csv
i = 1
for train_index, val_index in skf.split(np.zeros(len(data)),Y):
 training_data = data.iloc[train_index]
 validation_data = data.iloc[val_index]

 training_data.to_csv(f'/content/training_data_400x{i}.csv')
 validation_data.to_csv(f'/content/validation_data_400x{i}.csv')
 i+=1
