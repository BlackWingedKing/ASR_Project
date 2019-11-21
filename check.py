import re
import numpy as np
import tensorflow as tf
import os
# Retrieve weights from TF checkpoint
tf_path = os.path.abspath("/home/sanidhya/Sem7/ASR/Project_tf/multisensory/results/nets/shift/net.tf-650000")
init_vars = tf.train.list_variables(tf_path)[1:]
tf_vars = []
# print(init_vars)
for name, shape in init_vars:
	print("Loading TF weight {} with shape {}".format(name, shape))
	array = tf.train.load_variable(tf_path, name)
	tf_vars.append((name, array.squeeze()))
	print(name)

