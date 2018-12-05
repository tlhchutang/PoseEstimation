# Copyright 2018 Chu Tang (chu.tang@ninebot.com)
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

import io
import skimage
import logging
import os
from pycocotools.coco import COCO

from dataset_augment import pose_random_scale, pose_rotation, pose_flip, pose_resize_shortestedge_random, \
    pose_crop_random, pose_to_img
from dataset_prepare import CocoMetadata
import cv2
import numpy as np
import dataset_util

os.environ['CUDA_VISIBLE_DEVICES']='3'


root_path = "/data3_share_gpu1/coco/annotations/"
image_path_train = "/data3_share_gpu1/coco/train2017/"

image_path_val = "/data3_share_gpu1/coco/val2017/"
output_path = "/raid/tangc/coco_keypoints/"
num_kp = 17
crop_padding_x = 1.55
crop_padding_y = 1.55

model_input_width = 192
model_input_height = 192


TRAIN_JSON = "person_keypoints_train2017.json"
VAL_JSON = "person_keypoints_val2017.json"
TRAIN = True
def dict_to_tf_example(writer, image, kp_x, kp_y):
	height, width, _ = image.shape
	img_raw = np.asarray(image)
	img_raw = img_raw.tostring()
    #show images
	'''
	color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(0,0,0)]
	img_show = imgSinglePerson.resize((int(model_input_wh),int(model_input_wh)))
	tmpId = 0
	for cor in kp_cor_x:
		r = 4 #int(8/96.0*224)
		for ii in range(-r, r+1, 1):
			for jj in range(-r, r+1, 1):
				xxxx = cor + ii
				yyyy = kp_cor_y[tmpId] + jj
				if(xxxx < 0):
					xxxx = 0
				if(yyyy < 0):
					yyyy = 0
          		#if(xxxx>_show_img_w-1):
				#  xxxx=_show_img_w-1
				#if(yyyy>_show_img_h-1):
				#  yyyy=_show_img_h-1
          		img_show.putpixel((xxxx,yyyy),color[tmpId])

		tmpId += 1
	plt.imshow(img_show)
	plt.axis('on')
	plt.show()
	'''
	#print("kp coordinates = ", kp_x, kp_y)
	example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/encoded': dataset_util.bytes_feature(img_raw),
		'image/object/kp/kp_cor_x': dataset_util.int64_list_feature(kp_x),
		'image/object/kp/kp_cor_y': dataset_util.int64_list_feature(kp_y)
	}))
	writer.write(example.SerializeToString())

def get_bounding_box(xs, ys, vs):
	#get the tight bounding box by finding the max and min coordinates in x and y axis
	left = 100000
	top = 100000
	right = 0
	bottom = 0
	for idx in range(0, len(vs)):
		if (vs[idx] > 0):
			if left > xs[idx]:
				left = xs[idx]
			if right < xs[idx]:
				right = xs[idx]
			if top > ys[idx]:
				top = ys[idx]
			if bottom < ys[idx]:
				bottom = ys[idx]

	return left, top, right - left, bottom - top
			

def read_image( img_path):
    img_str = open(img_path, "rb").read()
    if not img_str:
        print("image not read, path=%s" % img_path)
    #nparr = np.fromstring(img_str, np.uint8)
    #return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

def crop_image(image, x, y, w, h, xs, ys, vs):
	origin_height, origin_width, channels = image.shape
	#padding
	center_x = x + w * 0.5
	center_y = y + h * 0.5
	left = center_x - 0.5 * crop_padding_x * w
	top = center_y - 0.5 * crop_padding_y *h
	right = center_x + 0.5 * crop_padding_x * w
	bottom = center_y + 0.5 * crop_padding_y * h

	#print("f t r b = ", left, top, right, bottom)   
 
	if left < 0:
		left = 0
	if top < 0:
		top = 0
	if right > origin_width - 1:
		right = origin_width - 1
	if bottom > origin_height - 1:
		bottom = origin_height - 1


	kp_list_x = []
	kp_list_y = []
	for idx in range(0, len(vs)):
		if vs[idx] > 0:
			kp_x = xs[idx] - left
			kp_y = ys[idx] - top
		else:
			kp_x = -1
			kp_y = -1
		kp_list_x.append(kp_x)
		kp_list_y.append(kp_y)

	return image[int(top):int(bottom), int(left):int(right)], kp_list_x, kp_list_y


def resize_image(image, kp_x, kp_y):
	h, w, _ = image.shape
	print ("image size = ", h , w)
	x_coeff = model_input_width * 1.0 / w
	y_coeff = model_input_height * 1.0 / h
	resized_x = []
	resized_y = []

	for idx in range(0, len(kp_x)):
		if kp_x[idx] < 0:
			resized_x.append(int(kp_x[idx]))
			resized_y.append(int(kp_y[idx]))
			continue
	
		new_x = kp_x[idx] * x_coeff
		new_y = kp_y[idx] * y_coeff
		resized_x.append(int(new_x))
		resized_y.append(int(new_y))
	return cv2.resize(image, (model_input_height, model_input_width)), resized_x, resized_y


def save_image(path, img, kp_x, kp_y, imgIdx, personIdx):
	for idx in range(0, len(kp_x)):
		if kp_x[idx] >= 0:
			cv2.circle(img, (kp_x[idx], kp_y[idx]), 4, (0,255,0), -1)
	img_name = path + "%s_%s.jpg" % (imgIdx, personIdx)
	cv2.imwrite(img_name, img)
 

def main(_):
	#print('hj')
	#raw_input()
	train_json_path = root_path + TRAIN_JSON
	val_json_path = root_path + VAL_JSON
	logging.info('Reading from %s .', train_json_path)

	train_annos = COCO(train_json_path)
	val_annos = COCO(val_json_path)

	if TRAIN:
		coco = train_annos
		tfrecord_path = output_path + "coco_train.record" 
		image_path = image_path_train
	else:
		coco = val_annos
		tfrecord_path = output_path + "coco_val.record"
		image_path = image_path_val


	writer = tf.python_io.TFRecordWriter(tfrecord_path)


	person_id = coco.getCatIds('person')
	imgIds = coco.getImgIds([], person_id)
	#image_person = coco.loadImgs(imgIds)
	#person_annotation_id = coco.getAnnIds(imgIds, person_id)
	#person_annotation = coco.loadAnns(person_annotation_id)

	path = "/raid/tangc/coco_keypoints/debug/"

	#imgIds = anno.getImgIds()
	total_person = 0
	for idx in range(0, len(imgIds)):
		print("processing image: %d / %d"  %(idx, len(imgIds)))
		anno_ids = coco.getAnnIds(imgIds[idx], person_id)
		annotations = coco.loadAnns(anno_ids)
		if(len(annotations) == 0):
			print("no annotations")
			continue
		categories = coco.loadCats(annotations[0]['category_id'])
		keypoints_name = coco.loadCats(annotations[0]['category_id'])[0]['keypoints']
		img_meta = coco.loadImgs(imgIds[idx])[0]
		img_path = os.path.join(image_path, img_meta['file_name'])
	
		img_width = int(img_meta['width'])
		img_height = int(img_meta['height'])

		image = read_image(img_path)
		print('image shape = ', image.shape)
		#if image.shape[2] != 3:
		#	input()
		if len(image.shape) < 3:
			continue
		
		num_person_per_image = 0
		for ann in annotations:
			if ann.get('num_keypoints', 0) == 0:
				continue

			kp = np.array(ann['keypoints'])
			xs = kp[0::3]
			ys = kp[1::3]
			vs = kp[2::3]

			x, y, w, h = get_bounding_box(xs, ys, vs)

			bbox = ann['bbox']

			#bbox get by keypoints smaller than detection bbox
			if w * h < bbox[2] * bbox[3]:
				x = int(bbox[0])
				y = int(bbox[1])
				w = int(bbox[2])
				h = int(bbox[3])

			if w < 60 or h < 80 or w * h < 3600:
				#print('down-top < 60 or right-left < 80 or (right-left)*(down-top) < 3600.')
				continue


			roi, kp_list_x, kp_list_y = crop_image(image, x, y, w, h, xs, ys, vs)	
			roi, kp_list_x, kp_list_y = resize_image(roi, kp_list_x, kp_list_y)
			num_person_per_image = num_person_per_image + 1
			dict_to_tf_example(writer, roi, kp_list_x, kp_list_y) 
			#for debug
			#if(idx % 1000 == 0):
			#	save_image(path, roi, kp_list_x, kp_list_y, str(idx), str(num_person_per_image))
	
		#print("number of person in current image = ", num_person_per_image)
		total_person = total_person + num_person_per_image
	print("total_person number = ", total_person)

if __name__ == '__main__':
	tf.app.run()

