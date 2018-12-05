import tensorflow as tf
from train_coco import generate_heatmap
import numpy as np

np.set_printoptions(threshold=np.nan)

a = tf.range(0, 10)
b = tf.range(0, 10)

w = 20
h = 20

inp = []
inp.append(h)
inp.append(w)
inp.append(a)
inp.append(b)

heatmap = tf.py_func(generate_heatmap, inp, tf.float32)



#test concat
norm = tf.constant([1, 2, 3, 4])
norm = tf.tile(norm, [6])
norm = tf.reshape(norm, [2, 3, 4])

concated = tf.concat(norm, 1)
with tf.Session() as sess:
	#print (sess.run(heatmap))
	print(sess.run(norm))
	print(sess.run(concated))
