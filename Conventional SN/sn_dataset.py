import subprocess
import pkg_resources

def install_required_packages():
    # List of required packages
    required_packages = ['spams', 'staintools']

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
        print("Packages 'spams' and 'staintools' are already installed.")

install_required_packages()
from tqdm import tqdm
import argparse
import staintools
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print("TensorFlow version:", tf.__version__)



init_varphi = np.asarray([[0.294, 0.110, 0.894],
                          [0.750, 0.088, 0.425]])

def acd_model(input_od, lambda_p=0.002, lambda_b=10, lambda_e=1, eta=0.6, gamma=0.5):
    """
    Stain matrix estimation by
    "Yushan Zheng, et al., Adaptive Color Deconvolution for Histological WSI Normalization."

    """
    alpha = tf.Variable(init_varphi[0], dtype='float32')
    beta = tf.Variable(init_varphi[1], dtype='float32')
    w = [tf.Variable(1.0, dtype='float32'), tf.Variable(1.0, dtype='float32'), tf.constant(1.0)]

    sca_mat = tf.stack((tf.cos(alpha) * tf.sin(beta), tf.cos(alpha) * tf.cos(beta), tf.sin(alpha)), axis=1)
    cd_mat = tf.matrix_inverse(sca_mat)

    s = tf.matmul(input_od, cd_mat) * w
    h, e, b = tf.split(s, (1, 1, 1), axis=1)

    l_p1 = tf.reduce_mean(tf.square(b))
    l_p2 = tf.reduce_mean(2 * h * e / (tf.square(h) + tf.square(e)))
    l_b = tf.square((1 - eta) * tf.reduce_mean(h) - eta * tf.reduce_mean(e))
    l_e = tf.square(gamma - tf.reduce_mean(s))

    objective = l_p1 + lambda_p * l_p2 + lambda_b * l_b + lambda_e * l_e
    target = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(objective)

    return target, cd_mat, w

class StainNormalizer(object):
    def __init__(self, pixel_number=100000, step=300, batch_size=1500):
        self._pn = pixel_number
        self._bs = batch_size
        self._step_per_epoch = int(pixel_number / batch_size)
        self._epoch = int(step / self._step_per_epoch)
        self._template_dc_mat = None
        self._template_w_mat = None

    def fit(self, images):
        opt_cd_mat, opt_w_mat = self.extract_adaptive_cd_params(images)
        self._template_dc_mat = opt_cd_mat
        self._template_w_mat = opt_w_mat

    def transform(self, images):
        if self._template_dc_mat is None:
            raise AssertionError('Run fit function first')

        opt_cd_mat, opt_w_mat = self.extract_adaptive_cd_params(images)
        transform_mat = np.matmul(opt_cd_mat * opt_w_mat,
                                  np.linalg.inv(self._template_dc_mat * self._template_w_mat))

        od = -np.log((np.asarray(images, float) + 1) / 256.0)
        normed_od = np.matmul(od, transform_mat)
        normed_images = np.exp(-normed_od) * 256 - 1

        return np.maximum(np.minimum(normed_images, 255), 0)

    def he_decomposition(self, images, od_output=True):
        if self._template_dc_mat is None:
            raise AssertionError('Run fit function first')

        opt_cd_mat, _ = self.extract_adaptive_cd_params(images)

        od = -np.log((np.asarray(images, float) + 1) / 256.0)
        normed_od = np.matmul(od, opt_cd_mat)

        if od_output:
            return normed_od
        else:
            normed_images = np.exp(-normed_od) * 256 - 1
            return np.maximum(np.minimum(normed_images, 255), 0)


    def sampling_data(self, images):
        pixels = np.reshape(images, (-1, 3))
        pixels = pixels[np.random.choice(pixels.shape[0], min(self._pn * 20, pixels.shape[0]))]
        od = -np.log((np.asarray(pixels, float) + 1) / 256.0)

        # filter the background pixels (white or black)
        tmp = np.mean(od, axis=1)
        od = od[(tmp > 0.3) & (tmp < -np.log(30 / 256))]
        od = od[np.random.choice(od.shape[0], min(self._pn, od.shape[0]))]

        return od

    def extract_adaptive_cd_params(self, images):
        """
        :param images: RGB uint8 format in shape of [k, m, n, 3], where
                       k is the number of ROIs sampled from a WSI, [m, n] is
                       the size of ROI.
        """
        od_data = self.sampling_data(images)
        input_od = tf.placeholder(dtype=tf.float32, shape=[None, 3])
        target, cd, w = acd_model(input_od)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for ep in range(self._epoch):
                for step in range(self._step_per_epoch):
                    sess.run(target, {input_od: od_data[step * self._bs:(step + 1) * self._bs]})
            opt_cd = sess.run(cd)
            opt_w = sess.run(w)
        return opt_cd, opt_w

def define_and_fit_normalizer(stain_normalization_technique, template_image_path):
    if stain_normalization_technique == 'R':
        normalizer = staintools.ReinhardColorNormalizer()
        normalizer.fit(staintools.read_image(template_image_path))
    elif stain_normalization_technique == 'M':
        normalizer = staintools.StainNormalizer(method="macenko")
        normalizer.fit(staintools.read_image(template_image_path))
    elif stain_normalization_technique == 'V':
        normalizer = staintools.StainNormalizer(method="vahadane")
        normalizer.fit(staintools.read_image(template_image_path))
    elif stain_normalization_technique == 'ACD':
        temp_image = cv.imread(template_image_path)
        temp_image_rgb = cv.cvtColor(temp_image, cv.COLOR_BGR2RGB)
        temp_image = np.asarray(temp_image_rgb)
        normalizer = StainNormalizer()
        normalizer.fit(temp_image)
    else:
        raise ValueError(f'Invalid stain normalization technique: {stain_normalization_technique}')


    return normalizer

def process_and_save_image(file_name, class_path, stain_normalized_dataset_path, type_folder, class_folder, sn_label, normalizer):
    image_path = os.path.join(class_path, file_name)

    # Check if the file is a valid image
    if not os.path.isfile(image_path) or not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f'Skipping non-image file: {file_name}')
        return

    # Apply the stain normalization transformation to the image
    if sn_label != 'ACD':
        transformed_image = normalizer.transform(staintools.read_image(image_path))
    else:
        im = cv.imread(image_path)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        source_img = np.asarray(im)
        transformed_image = normalizer.transform(source_img)

    # Save the transformed image in the corresponding directory in the stain-normalized dataset
    save_dir = os.path.join(stain_normalized_dataset_path, type_folder, class_folder)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file_name)
    cv.imwrite(save_path, transformed_image)


def main():
    technique_dict = {
    'R': 'Reinhard',
    'M': 'Macenko',
    'V': 'Vahadane',
    'ACD': 'ACD'
    }
    parser = argparse.ArgumentParser(description='Stain Normalization')
    parser.add_argument('--stain_normalization_technique', type=str, required=True, help='Stain normalization technique (R, M, V, or ACD)')
    parser.add_argument('--template_image_name', type=str, required=True, help='Name of the template image')
    parser.add_argument('--original_dataset_path', type=str, required=True, help='Path to the original dataset')
    parser.add_argument('--template_image_path', type=str, required=True, help='Path to the template folder')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory for the SN dataset')
    args = parser.parse_args()

    full_temp_img_path = f"{args.template_image_path}/{args.template_image_name}.png"
    # Create a directory to save the stain-normalized dataset
    stain_normalized_dataset_path = f'{args.save_dir}/{args.stain_normalization_technique}_{args.template_image_name}_FBCG_Dataset'
    os.makedirs(stain_normalized_dataset_path, exist_ok=True)

    print(f'You have selected {technique_dict[args.stain_normalization_technique]} Stain Normalization with {args.template_image_name}!!')
    print('Original dataset path: ',args.original_dataset_path)
    print('Template Folder Path:',args.template_image_path)
    print('Template Image path: ',full_temp_img_path)
    print('Saved SN Dataset path: ',stain_normalized_dataset_path)

    # Define a StainNormalizer object based on the chosen stain normalization technique and fit it to the template image
    normalizer = define_and_fit_normalizer(args.stain_normalization_technique, full_temp_img_path)

    # Loop over each folder in the base directory
    for type_folder in tqdm(os.listdir(args.original_dataset_path), desc='Processing folders'):
        type_path = os.path.join(args.original_dataset_path, type_folder)
        print(f'Processing folder: {type_path}')

        # Loop over each subfolder in the current folder
        for class_folder in tqdm(os.listdir(type_path), desc='Processing subfolders'):
            class_path = os.path.join(type_path, class_folder)
            print(f'Processing subfolder: {class_path}')

            # Loop over each file in the current subfolder
            for file_name in tqdm(os.listdir(class_path), desc='Processing files'):
                process_and_save_image(file_name, class_path, stain_normalized_dataset_path, type_folder, class_folder, args.stain_normalization_technique,normalizer)

if __name__ == "__main__":
    main()
