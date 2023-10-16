# Stain Normalization with deep_sn_dataset.py

This Python script performs stain normalization using the StainNet or StainGAN techniques on an input dataset. It allows you to preprocess and save the transformed images to a new directory.

## Prerequisites

Before using this script, make sure you have the following prerequisites installed:

- Python 3
- PyTorch
- OpenCV
- tqdm
- PIL (Pillow)

You can install the required Python packages using pip:

`!pip install torch opencv-python tqdm pillow`

## Usage

To use the script, follow these steps:

1. Clone or download the repository containing `deep_sn_dataset.py` to your local machine.

2. Navigate to the directory containing `deep_sn_dataset.py` in your terminal.

3. Run the script with the following command:

```
python deep_sn_dataset.py
--stain_normalization_technique [StainNet or StainGAN]
--checkpoint_path [Path to the checkpoint file]
--original_dataset_path [Path to the original dataset]
--save_dir [Path to the directory where the SN dataset will be saved]
```

- `--stain_normalization_technique`: Specify the stain normalization technique you want to use (StainNet or StainGAN).
- `--checkpoint_path`: Provide the path to the checkpoint file for the selected technique.
- `--original_dataset_path`: Set the path to the original dataset you want to perform stain normalization on.
- `--save_dir`: Specify the directory where the stain-normalized dataset will be saved.

4. The script will process the images in the original dataset, apply the stain normalization technique, and save the transformed images in the specified directory.

## Example

Here is an example command to run the script:

```bash
python deep_sn_dataset.py \
--stain_normalization_technique StainGAN \
--checkpoint_path "/path/to/latest_net_G_A.pth" \
--original_dataset_path "/path/to/original/dataset" \
--save_dir "/path/to/save/directory"
```

## To get StainGAN checkpoint path:
```
!wget https://github.com/khtao/StainNet/raw/94c20b31c0784d0d49468265afdde3d131d6afc8/checkpoints/camelyon16_dataset/latest_net_G_A.pth
```

Note

Make sure to have the correct checkpoint file for the selected technique.
The script assumes that your original dataset contains subfolders for different categories.
It will create a new directory for the stain-normalized dataset with the specified technique in the save_dir.
