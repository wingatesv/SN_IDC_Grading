import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import PIL
import cv2 as cv
from PIL import Image
import numpy as np
import functools
from torchvision.models import squeezenet1_1

class StainNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32, kernel_size=1):
        super(StainNet, self).__init__()
        model_list = []
        model_list.append(nn.Conv2d(input_nc, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
        model_list.append(nn.ReLU(True))
        for n in range(n_layer - 2):
            model_list.append(
                nn.Conv2d(n_channel, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
            model_list.append(nn.ReLU(True))
        model_list.append(nn.Conv2d(n_channel, output_nc, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))

        self.rgb_trans = nn.Sequential(*model_list)

    def forward(self, x):
        return self.rgb_trans(x)

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image=  image[np.newaxis, ...]
    image=  torch.from_numpy(image)
    return image

def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1,2,0))
    return image


def process_and_save_image(file_name, class_path, stain_normalized_dataset_path, type_folder, class_folder, sn_label, model):
    image_path = os.path.join(class_path, file_name)

    # Check if the file is a valid image
    if not os.path.isfile(image_path) or not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f'Skipping non-image file: {file_name}')
        return

    img_source = Image.open(image_path)

    # Apply the stain normalization transformation to the image
    if sn_label == 'StainNet':
      image_net=model(norm(img_source).cuda())
      transformed_image=un_norm(image_net)
    
    else:
      image_gan=model(norm(img_source).cuda())
      transformed_image=un_norm(image_gan)

    # Save the transformed image in the corresponding directory in the stain-normalized dataset
    save_dir = os.path.join(stain_normalized_dataset_path, type_folder, class_folder)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, file_name)
    cv.imwrite(save_path, transformed_image)


def main():

    parser = argparse.ArgumentParser(description='Stain Normalization')
    parser.add_argument('--stain_normalization_technique', type=str, required=True, help='Stain normalization technique (StainNet or StainGAN)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--original_dataset_path', type=str, required=True, help='Path to the original dataset')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory for the SN dataset')
    args = parser.parse_args()


    # Create a directory to save the stain-normalized dataset
    stain_normalized_dataset_path = f'{args.save_dir}/{args.stain_normalization_technique}_FBCG_Dataset'
    os.makedirs(stain_normalized_dataset_path, exist_ok=True)

    print(f'You have selected {args.stain_normalization_technique} Stain Normalization !!')
    print('Original dataset path: ',args.original_dataset_path)
    print('Saved SN Dataset path: ',stain_normalized_dataset_path)

    # Initialize StainNet
    if args.stain_normalization_technique == 'StainNet':
      model = StainNet().cuda()
      

    elif args.stain_normalization_technique == 'StainGAN':
      model = ResnetGenerator(3, 3, ngf=64, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9).cuda().cuda()


    else:
      raise ValueError(f'Invalid stain normalization technique: {args.stain_normalization_technique}')

    model.load_state_dict(torch.load(args.checkpoint_path))

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
                process_and_save_image(file_name, class_path, stain_normalized_dataset_path, type_folder, class_folder, args.stain_normalization_technique, model)

if __name__ == "__main__":
    main()
