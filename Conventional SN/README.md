# Convolutional Stain Normalization with conv_sn_dataset.py

This Python script is designed for stain normalization using convolutional neural networks (CNNs). It offers various stain normalization techniques, including Reinhard, Macenko, Vahadane, and ACD (Adaptive Color Deconvolution). The script can be used to preprocess and save transformed images from a given dataset.

## Prerequisites

Before using this script, ensure you have the following prerequisites installed:

- Python 3
- TensorFlow (compatible with version 1.x)
- OpenCV
- NumPy
- Pillow (PIL)
- tqdm

You can install the required Python packages using pip:

```bash
pip install tensorflow==1.15 opencv-python numpy pillow tqdm
```
##Usage

To use the script, follow these steps:

Clone or download the repository containing `conv_sn_dataset.py` to your local machine.
Navigate to the directory containing `conv_sn_dataset.py` in your terminal.
Run the script with the following command:

```bash
python conv_sn_dataset.py \
--stain_normalization_technique [R, M, V, or ACD] \
--template_image_name [Name of the template image] \
--original_dataset_path [Path to the original dataset] \
--template_image_path [Path to the template folder] \
--save_dir [Path to the directory where the SN dataset will be saved]
```

`--stain_normalization_technique`: Specify the stain normalization technique you want to use (R for Reinhard, M for Macenko, V for Vahadane, or ACD for Adaptive Color Deconvolution).
`--template_image_name`: Name of the template image (without the file extension).
`--original_dataset_path`: Set the path to the original dataset you want to perform stain normalization on.
`--template_image_path`: Specify the path to the folder containing the template image.
`--save_dir`: Specify the directory where the stain-normalized dataset will be saved.

The script will process the images in the original dataset, apply the selected stain normalization technique, and save the transformed images in the specified directory.

## Note
The script supports various stain normalization techniques:
R: Reinhard
M: Macenko
V: Vahadane
ACD: Adaptive Color Deconvolution

Make sure to provide a template image for the chosen stain normalization technique. The template image should be located in the specified template_image_path directory.
The script uses TensorFlow 1.x, so make sure you have the compatible version installed.
Ensure that the template image is in PNG format and that your original images are in RGB color format.
Example

Here is an example command to run the script:
```bash
python conv_sn_dataset.py \
--stain_normalization_technique V \
--template_image_name Temp5 \
--original_dataset_path "/path/to/original/dataset" \
--template_image_path "/path/to/template/folder" \
--save_dir "/path/to/save/directory"
```
