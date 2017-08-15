#!/usr/bin/python
# -*- coding: latin-1 -*-
# -------------------------- IMPORTS -------------------------- #
import numpy  as np
import random as rn
import scipy
import skimage.util
import skimage.transform

# -------------------------- CROP -------------------------- #
#
# image: image to be cropped, scale: scale factor, keep_original_size: keep image's original shape
def crop(image, scale=0.8, keep_original_size=True):
	'''
	Parameters:
		image: NumPy array of size NxMxC
		scale: float number between 0 and 1
		keep_original_size: boolean
	'''
	size_x, size_y, num_channels = image.shape
	if (scale < 1):
		max_x = int(round(size_x*scale))
		max_y = int(round(size_y*scale))
		dif_x = size_x - max_x
		dif_y = size_y - max_y
		bias_x = int(round(dif_x/2))
		bias_y = int(round(dif_y/2))
		image = image[bias_x:bias_x+max_x, bias_y:bias_y+max_y, :]
	if (keep_original_size):
		image = scipy.misc.imresize(image, (size_x,size_y,num_channels))
	return image

# -------------------------- METHOS TO RANDOMLY APPLY TRANSFORMATIONS -------------------------- #
def apply_random_rotation(image):
	'''
	Parameters:
		image: NumPy array of size NxMxC
	'''
	ang = rn.randint(0, 360)
	return rotate(n_image, angle=ang)

def apply_random_noise(image):
	'''
	Parameters:
		image: NumPy array of size NxMxC
	'''
	i = rn.randint(0,3)
	noise_mode = []
	noise_mode.append('gaussian')
	noise_mode.append('pepper')
	noise_mode.append('s&p')
	noise_mode.append('speckle')
	return skimage.util.random_noise(image, mode=noise_mode[i]) * 255

def apply_random_cropping(image):
	'''
	Parameters:
		image: NumPy array of size NxMxC
	'''
	i = rn.random()
	if (i >= 0.75): # Crop if the random scale is >= 75%, we don't want to lose too much info
		return crop(image, scale=i)
	else:
		return image

def apply_random_vertical_flip(image):
	'''
	Parameters:
		image: NumPy array of size NxMxC
	'''
	i = rn.random()
	if (i >= 0.5): # We do this to don't flip ALL the times (50/50 prob of flipping)
		return np.flipud(image) #cv.flip(image,1)
	else:
		return image

def apply_random_horizontal_flip(image):
	'''
	Parameters:
		image: NumPy array of size NxMxC
	'''
	i = rn.random()
	if (i >= 0.5): # We do this to flip just some very rare times (flip with 25% prob)
		return np.fliplr(image) #cv.flip(image,0)
	else:
		return image

# MAIN METHOD
def apply_random_transformations(image):
	'''
	This method applies all the transformations using the default parameters.
	Parameters:
		image: NumPy array of size NxMxC
	'''
	image = apply_random_noise(image)
	image = apply_random_cropping(image)
	image = apply_random_vertical_flip(image)
	image = apply_random_horizontal_flip(image)
	return image

def augment_single_data(image, labels, num_new):
	'''
	Augments a single image with its labels.
	Parameters:
		image: NumPy array of size NxMxC
		labels: integer that defines the class of the image
		num_new: integer that defines the number of new samples
	'''
	n = image.shape[0]
	m = image.shape[1]
	c = image.shape[2]
	new_images = np.empty([num_new, n, m, c], dtype=float)
	new_labels = np.zeros([num_new], dtype=int)
	for i in range(0, num_new):
		new_images[i,:,:,:] = apply_random_transformations(image)
		new_labels[i] = labels
	return new_images, new_labels


# -------------------------- Same as before but for one-hot encoding labels -------------------------- #
def augment_single_data_one_hot(image, labels, num_new):
	'''
	Augments a single image with its labels in one-hot encoding.
	Parameters:
		image: NumPy array of size NxMxC
		labels: NumPy array of size [1,N] -> label in one-hot encoding
		num_new: integer that defines the number of new samples
	'''
	n = image.shape[0]
	m = image.shape[1]
	c = image.shape[2]
	new_images = np.empty([num_new, n, m, c], dtype=float)
	new_labels = np.zeros([num_new, labels.shape[0]], dtype=int)
	for i in range(0, num_new):
		new_images[i,:,:,:] = apply_random_transformations(image)
		new_labels[i,:] = labels
	return new_images, new_labels

# -------------------------- Taking off the original dataset, and keep only the augmentated samples -------------------------- #
# Returns an augmented dataset without the original values
def augment_data_one_hot(dataset, labels, num_new):
	'''
	Augments dataset with its labels in one-hot encoding.
	Parameters:
		dataset: NumPy array of size ZxNxMxC. Thus, we have Z images of dimensions NxMxC.
		labels: NumPy array of size [1,N] -> label in one-hot encoding
		num_new: integer that defines the number of new samples
	'''
	n = dataset.shape[0]
	new_dataset = []
	new_labels  = []
	for i in range(n):
		feat, lab = augment_single_data_one_hot(dataset[i,:,:,:], labels[i,:], num_new, labels.shape[1])
		for c in range(feat.shape[0]):
			new_dataset.append(feat[c,:,:,:])
			new_labels.append(lab[c,:])
	return np.asarray(new_dataset), np.asarray(new_labels)

# -------------------------- METHODS TO BALANCE IMBALANCED DATASET -------------------------- #

def get_diff_binary_classes(target):
	'''
	Obtain the difference of number of instances between the two classes
	of the dataset. It only works for binary classes.
	'''
	unique, counts = np.unique(target, return_counts=True)
	z = dict(zip(unique, counts))
	majority_class_index = np.argmax(counts)
	minority_class_index = np.argmin(counts)
	difference = np.absolute(counts[0] - counts[1])
	return minority_class_index, majority_class_index, counts, difference

# This method balances an imbalanced dataset (usually the train set, although it
# can be used for an entire dataset). It only works for binary classes.
def balance_dataset(dataset, target):
	'''
	Balance an unbalanced dataset (usually the train set, although it
	can be used for the entire dataset). It only works for binary classes.
	'''
	# Obtain info about the classes (index of majority class, of minority class, etc.)
	min_class_idx, maj_class_idx, counts, difference = get_diff_binary_classes(target)
	print 'Num. of instances for each class before augmentation: ', counts
	# Augment each instance of the dataset (of the minority class)
	# until reaching the same number of instances of the majority class
	new_dataset = dataset
	new_target  = target
	counter = 0
	i = 0
	while (i < target.shape[0]):
		if (target[i] == min_class_idx):
			feature, label = augment_single_data(dataset[i,:,:,:], target[i], 1)
			new_dataset    = np.concatenate((new_dataset, feature), axis=0)
			new_target     = np.concatenate((new_target,  label),   axis=0)
			counter        = counter + 1
		if (i == target.shape[0]-1) and (counter != difference):
			i = 0
		if counter == difference:
			break
		i = i + 1
	# Check if train and trainl are balanced
	min_class_idx, maj_class_idx, counts, difference = get_diff_binary_classes(new_target)
	print 'Num. of instances for each class after augmentation: ', counts
	# Return the balanced dataset
	return new_dataset, new_target
#
