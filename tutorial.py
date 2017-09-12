#!/usr/bin/python
# -------------------------- IMPORTS -------------------------- #
import numpy                    as np
import augment	 				as ag
import parallel_augment 		as pa

# -------------------------- LOAD DATASET -------------------------- #
'''
Here we load the dataset.
The functions we provide receive as input the dataset and the labels in NumPy array formats with the following shapes:

dataset -> (num_samples, im_size_x, im_size_y, num_channels) for all functions
labels -> (num_samples) for the augment.balance_dataset function
labels -> (num_samples, num_classes), which is ONE-HOT encoding, for the parallel_augment.augment_dataset_parallel function

Values of dataset must be in the range [0,255]

'''

# We first load our dataset
dataset, labels = load_dataset() # You should implement this


# -------------------------- BALANCE DATASET -------------------------- #
# In most cases, we'll probably have an unbalanced dataset, so if
# you want to balance the classes of the dataset through data augmentation,
# we use:
dataset, labels = ag.balance_dataset(dataset=dataset, target=labels)
# For this function, use labels with shape: (num_samples)
# This function works only with two classes that are binary


# -------------------------- AUGMENT DATASET -------------------------- #
# For the following functions we use the one-hot encoding for the labels,
# thus we convert the labels to that encoding through:
num_classes = 2 # We have two classes
labels  = (np.arange(num_classes) == labels[:,None]).astype(np.float32)

# Once the labels array has the one-hot encoding format, we can augment
# the dataset sequentially or in parallel.
# Say we want to augment the dataset 10 times:
num_new = 10

# Sequential augmentation
new_dataset, new_labels = ag.augment_data_one_hot(dataset=dataset, labels=labels, num_new=num_new)

# Parallel
# For this step, the parallel python library is necessary
num_cpus = 4 # Set number of CPUs to use
servers_list = ('localhost,') # List of servers to augment using a cluster, here we use only the localhost
template = pa.get_pp_template(num_cpus=num_cpus, servers_list=servers_list) # Get a template
new_dataset, new_labels = pa.augment_dataset_parallel(dataset=dataset, labels=labels, template=template, num_aug=num_new)

# -------------------------- TRANSFORMATIONS -------------------------- #
'''
If you want to use transformations individually you can call the following functions:

ag.crop
ag.apply_random_rotation
ag.apply_random_noise
ag.apply_random_cropping
ag.apply_random_vertical_flip
ag.apply_random_horizontal_flip

If you want to apply a combination of all transformations on a single sample you
can use:

ag.augment_single_data or ag.augment_single_data_one_hot, depending if your label
is in one-hot encoding or not.

You can find more detailed info on how to use these functions in the augment.py file.

'''

#
