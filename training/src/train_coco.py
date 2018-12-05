# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import time
import numpy as np
import configparser
#import dataset

from datetime import datetime

#from dataset import get_train_dataset_pipeline, get_valid_dataset_pipeline
from networks import get_network
from dataset_prepare import CocoPose
from dataset_augment import set_network_input_wh, set_network_scale
import math
import cv2

r_mean = 118.06
g_mean = 113.75
b_mean = 106.56
norm_scale = 0.176

'''
def get_input(batchsize, epoch, is_train=True):
    if is_train is True:
        input_pipeline = get_train_dataset_pipeline(batch_size=batchsize, epoch=epoch, buffer_size=100)
    else:
        input_pipeline = get_valid_dataset_pipeline(batch_size=batchsize, epoch=epoch, buffer_size=100)
    iter = input_pipeline.make_one_shot_iterator()
    _ = iter.get_next()
    return _[0], _[1]
'''

def get_loss_and_output(model, batchsize, input_image, input_heat, reuse_variables=None):
    losses = []

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        _, pred_heatmaps_all = get_network(model, input_image, True)

    #stage by stage, N * H * W * C per stage
    for idx, pred_heat in enumerate(pred_heatmaps_all):
        #pred_heat size N * H * W * C
        #print(idx)
        #print(pred_heat.get_shape())
        loss_l2 = tf.nn.l2_loss(tf.concat(pred_heat, axis=0) - input_heat, name='loss_heatmap_stage%d' % idx)
        losses.append(loss_l2)

    total_loss = tf.reduce_sum(losses) / batchsize
    total_loss_ll_heat = tf.reduce_sum(loss_l2) / batchsize
    return total_loss, total_loss_ll_heat, pred_heat


def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

keys_to_features = {
	'image/height': 
		tf.FixedLenFeature((), tf.int64, 1),
	'image/width':
		tf.FixedLenFeature((), tf.int64, 1),
 	'image/encoded':
		tf.FixedLenFeature((), tf.string, default_value=''),
 	'image/object/kp/kp_cor_x':
		tf.VarLenFeature(tf.int64),
 	'image/object/kp/kp_cor_y': 
		tf.VarLenFeature(tf.int64)
  }


def put_heatmap(heatmap, plane_idx, center, sigma):
	center_x, center_y = center
	_, height, width = heatmap.shape[:3]

	th = 1.6052
	delta = math.sqrt(th * 2)

	x0 = int(max(0, center_x - delta * sigma))
	y0 = int(max(0, center_y - delta * sigma))

	x1 = int(min(width, center_x + delta * sigma))
	y1 = int(min(height, center_y + delta * sigma))

	# gaussian filter
	for y in range(y0, y1):
		for x in range(x0, x1):
			d = (x - center_x) ** 2 + (y - center_y) ** 2
			exp = d / 2.0 / sigma / sigma
			if exp > th:
				continue
			heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
			heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)



target_size = 96
input_height = 192
input_width = 192
def generate_heatmap(height, width, dense_tensor_x, dense_tensor_y):
	'''
	heatmap = np.zeros((17, height, width), dtype=np.float32)
	for idx in range(0, len(dense_tensor_x)):
		if dense_tensor_x[idx] < 0 or dense_tensor_y[idx] < 0:
				continue
		put_heatmap(heatmap, idx, (dense_tensor_x[idx], dense_tensor_y[idx]), 6.0)
		
	heatmap = heatmap.transpose((1, 2, 0))
	# background
	# heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)
	if target_size:
		heatmap = cv2.resize(heatmap, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
	'''
	#directly generate resized heatmap
	heatmap = np.zeros((17, target_size, target_size), dtype=np.float32)
	resize_coeffient = target_size * 1.0 / input_height

	for idx in range(0, len(dense_tensor_x)):
		if dense_tensor_x[idx] < 0 or dense_tensor_y[idx] < 0:
				continue
		put_heatmap(heatmap, idx, (dense_tensor_x[idx] * resize_coeffient, dense_tensor_y[idx] * resize_coeffient), 3.0)
		
	heatmap = heatmap.transpose((1, 2, 0))
	# background
	# heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)
	if target_size:
		heatmap = cv2.resize(heatmap, (target_size, target_size), interpolation=cv2.INTER_AREA)




	return_heatmap = np.copy(heatmap)
	return return_heatmap.astype(np.float32)

def decode_record(filename_queue):
	with tf.name_scope('decode_record'):
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
				serialized_example,
				keys_to_features)
		image = tf.decode_raw(features['image/encoded'], tf.uint8)
		#print(type(features['image/height']))
		height = tf.cast(features['image/height'], tf.int32)
		width = tf.cast(features['image/width'], tf.int32)

		#image = tf.reshape(image, [raw_height, raw_width, 3])

		image = tf.reshape(image, [input_height, input_width, 3])
		image = tf.cast(image, tf.float32)
		print(image.get_shape())
		dense_tensor_x = tf.sparse_tensor_to_dense(features['image/object/kp/kp_cor_x'])
		dense_tensor_y = tf.sparse_tensor_to_dense(features['image/object/kp/kp_cor_y'])
		#generate_heatmap(height, width, dense_tensor_x, dense_tensor_y)
		inp = []
		inp.append(input_height)
		inp.append(input_width)
		inp.append(dense_tensor_x)
		inp.append(dense_tensor_y)

		heatmap = tf.py_func(generate_heatmap, inp, tf.float32)
		#heatmap = tf.random.normal([96,96,17])
		#image = tf.random.normal([192,192,3])
		'''
        if RANDOM_CROP_FLIP:
            image, offset_x, offset_y, crop_width, crop_height, flip = random_crop(image)
        else:
            #center crop to width [height, 1872, 3]
            begin = [0, 9, 0]
            size = [height, 2000, 3]
            image = tf.slice(image, begin, size)

        image = tf.image.resize_images(image, [resize_height, resize_width]) # w ,h or h, w? 
        image = tf.cast(image, tf.float32)
        image = (image - [[[R_MEAN, G_MEAN, B_MEAN]]])*SCALE
   
        #convert stixel position to ground truth map
        #prediction shape is 1 * 100 * 74
    
        #stixel_position = tf.decode_raw(features['image/stixel_position'], tf.uint8)
        stixel_position = tf.sparse_tensor_to_dense(features['image/stixel_position']) 
        stixel_position = tf.reshape(stixel_position, [1, 2028])
        stixel_position = tf.cast(stixel_position, dtype = tf.float64)
        #normalize to [0 ~ 1]
        #normalize_factor = tf.divide(1, height)
        #stixel_position = tf.multiply(stixel_position, normalize_factor)

        stixel_position = tf.reshape(stixel_position, [2028])
        prediction_shape = [output_height, output_width, output_channel]
    
    
        if RANDOM_CROP_FLIP:
            #do corresponding stixel label crop and flip
            stixel_position = random_crop_stixel_position(stixel_position, offset_x, offset_y, crop_width, crop_height, flip)
            print("*************************")
            print(stixel_position)    
        else:
            #normalize to [0 ~ 1]
            normalize_factor = tf.divide(1, height)
            stixel_position = tf.multiply(stixel_position, normalize_factor)
            stixel_begin = [9]
            stixel_size = [2000]
            stixel_position = tf.slice(stixel_position, stixel_begin, stixel_size)
        
        if USE_LOCAL_PROB:
            stixel_gt = convert_to_sparse_stixel_position(stixel_position, resize_height, resize_width, prediction_shape)
        else:
            stixel_gt = convert_to_prediction_shape(stixel_position, resize_height, resize_width, prediction_shape)
		'''

	
		#print(heatmap.get_shape())
		#image.set_shape([input_height, input_width, 3])

		heatmap.set_shape([target_size, target_size, 17])
		heatmap = tf.cast(heatmap, tf.float32)
		#image = tf.cast(image, tf.float32)
		#zero mean image in b,g,r order
		image = (image - [[[b_mean, g_mean, r_mean]]]) * norm_scale
		return image, heatmap


def fetch_data_batch(filename, batch_size, num_epochs=None):
    '''num_features := width * height for 2D image'''
    print(filename)
    filename_queue = tf.train.string_input_producer(
            filename, shuffle=True)
    #filename, num_epochs = num_epochs, shuffle=True)

    example, label = decode_record(filename_queue)
   
     
    #print('read_decode done.')
    min_after_dequeue = 64
    capacity = min_after_dequeue + 20 * batch_size
   
    example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, num_threads = 40, capacity=capacity,
            min_after_dequeue=min_after_dequeue)

    print("batch size = %d" % batch_size)
    #example_batch, label_batch = tf.train.batch(
    #        [example, label], batch_size = batch_size, num_threads = 20,
    #        capacity = capacity)

    return example_batch, label_batch





def main(argv=None):
    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config_file = "experiments/mv2_cpm.cfg"
    if len(argv) != 1:
        config_file = argv[1]
    config.read(config_file)
    for _ in config.options("Train"):
        params[_] = eval(config.get("Train", _))

    os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']
    gpus_index = params['visible_devices'].split(",")
    params['gpus'] = len(gpus_index)

    if not os.path.exists(params['modelpath']):
        os.makedirs(params['modelpath'])
    if not os.path.exists(params['logpath']):
        os.makedirs(params['logpath'])

    #dataset.set_config(params)
    set_network_input_wh(params['input_width'], params['input_height'])
    set_network_scale(params['scale'])

    training_name = '{}_batch-{}_lr-{}_gpus-{}_{}x{}_{}'.format(
        params['model'],
        params['batchsize'],
        params['lr'],
        params['gpus'],
        params['input_width'], params['input_height'],
        config_file.replace("/", "-").replace(".cfg", "")
    )

    input_height = params['input_height']
    input_width = params['input_width']
    target_size = params['input_height'] // params['scale']
    r_mean = params['r_mean']
    g_mean = params['g_mean']
    b_mean = params['b_mean']
    norm_scale = params['norm_scale']

    #with tf.Graph().as_default(), tf.device("/cpu:0"):
    with tf.Graph().as_default():
        #input_image, input_heat = get_input(params['batchsize'], params['max_epoch'], is_train=True)
        #valid_input_image, valid_input_heat = get_input(params['batchsize'], params['max_epoch'], is_train=False)
        input_image_batch, input_heat_batch = fetch_data_batch([params['training_data']], params['batchsize'])
        valid_input_image_batch, valid_input_heat_batch = fetch_data_batch([params['validation_data']], params['batchsize'])
       
        input_image = tf.placeholder(tf.float32, shape=[None, params['input_height'], params['input_width'], 3])
        input_heat = tf.placeholder(tf.float32, shape=[None, target_size, target_size, params['n_kpoints']])
        
        #val_input_image = tf.placeholder(tf.float32, shape=[None, params['input_height'], params['input_width'], 3])
        #val_input_heat = tf.placeholder(tf.float32, shape=[None, target_size, target_size, params['n_kpoints']])



        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(float(params['lr']), global_step,
                                                   decay_steps=10000, decay_rate=float(params['decay_rate']), staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        tower_grads = []
        reuse_variable = False
        # multiple gpus
        '''
        for i in range(params['gpus']):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("GPU_%d" % i):
                    loss, last_heat_loss, pred_heat = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat, reuse_variable)
                    reuse_variable = True
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

                    #valid_loss, valid_last_heat_loss, valid_pred_heat = get_loss_and_output(params['model'], params['batchsize'],
                    #                                                                        valid_input_image, valid_input_heat, reuse_variable)
                    valid_loss, valid_last_heat_loss, valid_pred_heat = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat, reuse_variable)
        '''


        loss, last_heat_loss, pred_heat = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat, reuse_variable)
        reuse_variable = True
        grads = opt.compute_gradients(loss)
        tower_grads.append(grads)
        valid_loss, valid_last_heat_loss, valid_pred_heat = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat, reuse_variable)
 

        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(max_to_keep=100)

        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss_lastlayer_heat", last_heat_loss)
        summary_merge_op = tf.summary.merge_all()

        pred_result_image = tf.placeholder(tf.float32, shape=[params['val_batchsize'], 480, 640, 3])
        pred_result__summary = tf.summary.image("pred_result_image", pred_result_image, params['val_batchsize'])

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])
            print("Start training...")
            for step in range(total_step_num):
                start_time = time.time()
                
                #fetch training data
                images, labels = sess.run([input_image_batch, input_heat_batch])
                fetch_time = time.time() - start_time
                #print(step)
                #print(images)
                #print(labels)
                feed_dict = {
                    input_image: images,
                    input_heat: labels
                }

                #_, loss_value, lh_loss, in_image, in_heat, p_heat = sess.run(
                #    [train_op, loss, last_heat_loss, input_image, input_heat, pred_heat]
                #)
                _, merge_op, loss_value, lh_loss, p_heat = sess.run(
                    [train_op, summary_merge_op, loss, last_heat_loss, pred_heat], feed_dict = feed_dict)
               
                duration = time.time() - start_time



                
                if step != 0 and step % params['per_update_tensorboard_step'] == 0:
                    # False will speed up the training time.
                    if params['pred_image_on_tensorboard'] is True:

                        #valid_loss_value, valid_lh_loss, valid_in_image, valid_in_heat, valid_p_heat = sess.run(
                        #    [valid_loss, valid_last_heat_loss, valid_input_image, valid_input_heat, valid_pred_heat]
                        #)

                           
                        val_images, val_labels = sess.run([valid_input_image_batch, valid_input_heat_batch])
                        val_feed_dict = {
                           input_image: val_images,
                           input_heat: val_labels
                        }
                        valid_loss_value, valid_lh_loss, valid_p_heat = sess.run(
                           [valid_loss, valid_last_heat_loss, valid_pred_heat], feed_dict = val_feed_dict)

                        result = []
                        for index in range(params['val_batchsize']):
                            r = CocoPose.display_image(
                                    val_images[index,:,:,:],
                                    val_labels[index,:,:,:],
                                    valid_p_heat[index,:,:,:],
                                    True
                                )
                            result.append(
                                r.astype(np.float32)
                            )

                        comparsion_of_pred_result = sess.run(
                            pred_result__summary,
                            feed_dict={
                                pred_result_image: np.array(result)
                            }
                        )
                        summary_writer.add_summary(comparsion_of_pred_result, step)
                        

                # print train info
                num_examples_per_step = params['batchsize'] * params['gpus']
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / params['gpus']
                format_str = ('%s: step %d, loss = %.2f, last_heat_loss = %.2f (%.1f examples/sec; %.3f sec/batch; %.3f sec/fetch_batch)')
                print(format_str % (datetime.now(), step, loss_value, lh_loss, examples_per_sec, sec_per_batch, fetch_time))

                # tensorboard visualization
                #merge_op = sess.run(summary_merge_op)
                summary_writer.add_summary(merge_op, step)

                # save model
                if step % params['per_saved_model_step'] == 0:
                    checkpoint_path = os.path.join(params['modelpath'], training_name, 'model')
                    saver.save(sess, checkpoint_path, global_step=step)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
