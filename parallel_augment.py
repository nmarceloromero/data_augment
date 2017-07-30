#!/usr/bin/python
# -*- coding: latin-1 -*-
# ----------------------- IMPORTS ----------------------- #
import numpy as np
from augment import *

# ----------------------- PARALLEL FUNCTION TO AUGMENT ----------------------- #
#
def get_pp_template(num_cpus=1, servers_list=('localhost',)):
	'''
	Method to set up the template.
	Parameters:
		num_cpus: number of CPUs to be used.
		servers_list: tuple containing the list of available servers.
	'''
	# Import PP library
	import pp

	# Start servers
	ppservers = (servers_list) # For more servers: ("localhost", "192.168.0.1", "192.168.0.2", "192.168.0.3")

	# Number of workers to be used
	if num_cpus > 1:
		job_server = pp.Server(num_cpus, ppservers=ppservers)
	else:
		job_server = pp.Server(ppservers=ppservers) # Create server with automatically detected num of workers
		ncpus = job_server.get_ncpus()

	print 'Starting PP with: ', job_server.get_ncpus(), "workers..."

	# Start template
	template = pp.Template(	job_server 	= job_server,
							func       	= augment_single_data_one_hot,
							depfuncs	= (	apply_random_transformations,
										apply_random_noise,
										apply_random_swirl,
										apply_random_cropping,
										apply_random_vertical_flip,
										apply_random_horizontal_flip,
										crop),
							modules 	= ('numpy as np', 'random as rn', 'skimage.util', 'skimage.transform', 'scipy'))

	return template


#
def augment_dataset_parallel(dataset, labels, template, num_aug):
	'''
	Applies the template to augment dataset and labels in parallel.
	Labels must be in one-hot encoding.
	'''
	new_dataset = []
	new_labels  = []
	i = 0
	while (i < labels.shape[0]):
		jobs = []
		for c in range(template.job_server.get_ncpus()):
			print 'Augmenting sample ', i, '...'
			jobs.append(template.submit(dataset[i,:,:,:], labels[i,:], num_aug))
			i += 1
			if (i == labels.shape[0]):
				break
		for c in range(len(jobs)):
			new_d, new_l = jobs[c]()
			new_dataset.extend(new_d)
			new_labels.extend(new_l)
	return np.asarray(new_dataset), np.asarray(new_labels)


#
