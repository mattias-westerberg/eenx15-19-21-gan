import numpy as np
import scipy.misc
import os

print("util.py's __name__: {}".format(__name__))
print("util.py's __package__: {}".format(__package__))

TRANSFORM_CROP = "crop"
TRANSFORM_RESIZE = "resize"
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def is_pow2(x):
    return x & (x - 1) == 0

def get_paths(directory):
    """
    Produces a list of supported filetypes from the specified directory.

    INPUTS
        directory:    the directory path containing the files

    RETURNS
        - a list of the files
    """
    paths = [os.path.join(directory, f) for f in os.listdir(directory) if any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS)]
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
def merge(images, size):
    """ Takes a set of 'images' and creates an array from them.

    INPUT
        images:     the set of input images
        size:       [height, width] of the array

    RETURNS
        - image array as a single image
    """ 
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img
    
#ARRAY TO IMAGE FUNCTION
def imsave(images, size, path):
    """ Takes a set of `images` and calls the merge function. Converts
    the array to image data.

    INPUT
        images: the set of input images
        size:   [height, width] of the array
        path:   the save location

    RETURNS
        - an image array
    """
    img = merge(images, size)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return scipy.misc.imsave(path, (255 * img).astype(np.uint8))

#SAVE IMAGE FUNCTION
def save_images(images, size, image_path):
    """ takes an image and saves it to disk. Redistributes
    intensity values [-1 1] from [0 255]
    """
    return imsave(inverse_transform(images), size, image_path)

#INVERSE TRANSFORMATION OF INTENSITITES
def inverse_transform(images):
    """ This turns the intensities from [-1 1] to [0 1]

    INPUTS
        images:    the image to be transformed

    RETURNS
        -the transformed image
    """
    return (images + 1.0) / 2.