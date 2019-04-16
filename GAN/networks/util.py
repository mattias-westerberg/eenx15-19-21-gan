import numpy as np
import scipy.misc
import os
import re
import glob

TRANSFORM_CROP = "crop"
TRANSFORM_RESIZE = "resize"
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def load_data(file_name):
    """
    Return a dictionary with the form { image_path : [[x0 y0 x1 y1 c]] }
    """
    dir_name = os.path.dirname(file_name)
    train_dict = dict()
    train_imgs = []  # useful to keep an ordering of the imgs
    with open(file_name, "r") as file:
        for line in file:
            info = re.split(",| ", line[:-1])  # the last character is '\n'
            img_path = dir_name + "/" + info[0]
            train_imgs.append(img_path)
            bounding_boxes = []  # with the label
            for i in range(1, len(info), 5):
                bounding_boxes.append([int(value) for value in info[i:i + 5]])

            train_dict[img_path] = bounding_boxes

    return train_dict

def is_pow2(x):
    return x & (x - 1) == 0

def resize_bounding_boxes(bounding_boxes, new_size):
    res = {}
    for path, val in bounding_boxes.items():
        image = imread(path)
        xr = new_size / image.shape[0]
        yr = new_size / image.shape[1]
        
        # Clamp values to [0, new_size) for the bounding box loss function to work.
        x0 = max(0, min(xr * val[0], new_size - 1))
        y0 = max(0, min(yr * val[1], new_size - 1)) 
        x1 = max(0, min(xr * val[2], new_size - 1))
        y1 = max(0, min(yr * val[3], new_size - 1))
        c = val[4]
        res[path] = [x0, y0, x1, y1, c]

    return res

def get_paths(directory):
    """
    Produces a list of supported filetypes from the specified directory.

    INPUTS
        directory:    the directory path containing the files

    RETURNS
        - a list of the files
    """
    
    paths = []
    for f in glob.iglob(directory + '**/**', recursive=True):
        if os.path.isfile(f):
            if any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                paths.append(f)

    return paths


def get_image(image_path, image_size, input_transform=TRANSFORM_RESIZE):
    """ Loads the image and transforms it to 'image_size'

    Args:
        input_transform:    the method used to reshape the image
        image_path: location of the image
        image_size: size (in pixels) of the output image

    Returns:
        the cropped image
    """
    return transform(imread(image_path), image_size, input_transform)

def imread(path):
    """ Reads in the image (part of get_image function)"""
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

# TRANSFORM/CROPPING WRAPPER
def transform(image, npx=64, input_transform=TRANSFORM_RESIZE):
    """ Transforms the image by cropping and resizing and 
    normalises intensity values between -1 and 1

    INPUT
        image:      the image to be transformed
        npx:        the size of the transformed image [npx x npx]
        is_crop:    whether to preform cropping too [True or False]

    RETURNS
        - the transformed, normalised image
    """
    if input_transform == TRANSFORM_CROP:
        output = center_crop(image, npx)
    elif input_transform == TRANSFORM_RESIZE:
        output = scipy.misc.imresize(image, (npx, npx), interp='bicubic')
    else:
        output = image
    return np.array(output) / 127.5 - 1.0

# IMAGE CROPPING FUNCTION
def center_crop(x, crop_h, crop_w=None, resize_w=64):
    """ Crops the input image at the centre pixel

    INPUTS
        x:      the input image
        crop_h: the height of the crop region
        crop_w: if None crop width = crop height
        resize_w: the width of the resized image

    RETURNS
        - the cropped image
    """
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])

#CREATE IMAGE ARRAY FUNCTION
def merge(images, shape):
    """ Takes a set of 'images' and creates an array from them.

    INPUT
        images:     the set of input images
        shape:       [height, width] of the array

    RETURNS
        - image array as a single image
    """ 
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * shape[0]), int(w * shape[1]), 3))
    for idx, image in enumerate(images):
        i = idx % shape[1]
        j = idx // shape[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img
    
#ARRAY TO IMAGE FUNCTION
def imsave(images, paths):
    """ Takes a set of `images` and calls the merge function. Converts
    the array to image data.

    INPUT
        images: the set of input images
        size:   [height, width] of the array
        path:   the save location

    RETURNS
        - an image array
    """
    
    for img, pth in zip(images, paths):
        directory = os.path.dirname(pth)
        if not os.path.exists(directory):
            os.makedirs(directory)
        scipy.misc.imsave(pth, img)

#SAVE IMAGE FUNCTION
def save_images(images, paths):
    """ takes an image and saves it to disk. Redistributes
    intensity values [-1 1] from [0 255]
    """
    imgs = inverse_transform(images)
    imgs = (255 * imgs).astype(np.uint8)
    return imsave(imgs, paths)

#SAVE IMAGE FUNCTION
def save_mosaic(images, shape, path):
    """ takes an image and saves it to disk. Redistributes
    intensity values [-1 1] from [0 255]
    """
    imgs = inverse_transform(images)
    imgs = (255 * imgs).astype(np.uint8)
    img = merge(imgs, shape)
    return imsave(img, [path])

#INVERSE TRANSFORMATION OF INTENSITITES
def inverse_transform(images):
    """ This turns the intensities from [-1 1] to [0 1]

    INPUTS
        images:    the image to be transformed

    RETURNS
        -the transformed image
    """
    return (images + 1.0) / 2.